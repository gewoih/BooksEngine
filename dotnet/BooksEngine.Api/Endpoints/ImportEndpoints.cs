using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Import;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class ImportEndpoints
{
    public static void MapImport(this RouteGroupBuilder api)
    {
        api.MapPost("/me/import", async (IFormFile file, ClaimsPrincipal user, NpgsqlDataSource ds) =>
        {
            CsvFormat format;
            List<CsvRow> rows;
            try { (format, rows) = RatingsCsv.Parse(file.OpenReadStream()); }
            catch (FormatException e) { return Results.BadRequest(e.Message); }

            var errors = rows.Where(r => r.Error is not null).Select(r => new ImportIssue(r.Line, r.Label, r.Error!)).ToList();
            var skipped = rows.Count(r => r.Error is null && r.Rating == 0);           // Goodreads: 0 — без оценки
            var valid = rows.Where(r => r.Error is null && r.Rating > 0).ToList();
            var notFound = valid.Where(r => r.ExternalId.Length == 0)
                .Select(r => new ImportIssue(r.Line, r.Label, "нет id")).ToList();
            var ids = valid.Where(r => r.ExternalId.Length > 0).Select(r => r.ExternalId).Distinct().ToArray();

            await using var c = await ds.OpenConnectionAsync();
            // Наш формат: тень → главное (work_merges), иначе work. Goodreads: издание → его произведение, и если
            // оно тень — главное: издания теней в каталоге остаются при своём произведении (вне ядра)
            var sql = format == CsvFormat.Ours
                ? """
                  SELECT i.ext, w.id AS work_id, w.in_cf
                  FROM unnest(@ids) AS i(ext)
                  LEFT JOIN work_merges m ON m.shadow_external_id = i.ext
                  LEFT JOIN external_ids x ON x.source_id = (SELECT id FROM sources WHERE code = 'goodreads')
                                           AND x.entity_type = 'work' AND x.external_id = i.ext
                  JOIN works w ON w.id = coalesce(m.main_work_id, x.internal_id)
                  """
                : """
                  SELECT i.ext, w.id AS work_id, w.in_cf
                  FROM unnest(@ids) AS i(ext)
                  JOIN external_ids x ON x.source_id = (SELECT id FROM sources WHERE code = 'goodreads')
                                      AND x.entity_type = 'edition' AND x.external_id = i.ext
                  JOIN editions e ON e.id = x.internal_id
                  LEFT JOIN external_ids xw ON xw.source_id = x.source_id AND xw.entity_type = 'work'
                                            AND xw.internal_id = e.work_id
                  LEFT JOIN work_merges m ON m.shadow_external_id = xw.external_id
                  JOIN works w ON w.id = coalesce(m.main_work_id, e.work_id)
                  """;
            var found = (await c.QueryAsync<(string Ext, long WorkId, bool InCf)>(sql, new { ids }))
                .ToDictionary(f => f.Ext);
            notFound.AddRange(valid.Where(r => r.ExternalId.Length > 0 && !found.ContainsKey(r.ExternalId))
                .Select(r => new ImportIssue(r.Line, r.Label, "нет в каталоге")));

            // Несколько строк одного произведения (тень и главное) — средняя, округлённая как metrics.rounded;
            // «бросил» — если все строки книги dnf (как recommend.read_profile)
            var perWork = valid.Where(r => found.ContainsKey(r.ExternalId))
                .GroupBy(r => found[r.ExternalId].WorkId)
                .Select(g => (WorkId: g.Key, Value: (int)Math.Floor(g.Average(r => r.Rating!.Value) + 0.5),
                              Dnf: g.All(r => r.Dnf), First: g.First()))
                .ToList();
            var outside = perWork.Where(p => !found[p.First.ExternalId].InCf)
                .Select(p => new ImportIssue(p.First.Line, p.First.Label, "вне CF-ядра — не влияет на рекомендации")).ToList();

            var userId = user.UserId();
            var existing = (await c.QueryAsync<long>("SELECT work_id FROM ratings WHERE user_id = @userId AND work_id = ANY(@w)",
                new { userId, w = perWork.Select(p => p.WorkId).ToArray() })).ToHashSet();
            await using var tx = await c.BeginTransactionAsync();
            foreach (var p in perWork) await RatingsEndpoints.UpsertAsync(c, tx, userId, p.WorkId, p.Value, p.Dnf);
            await tx.CommitAsync();

            return Results.Ok(new ImportResultDto(perWork.Count(p => !existing.Contains(p.WorkId)),
                perWork.Count(p => existing.Contains(p.WorkId)), skipped, notFound.OrderBy(i => i.Line).ToList(),
                errors, outside));
        }).RequireAuthorization().DisableAntiforgery().Produces<ImportResultDto>();
    }
}
