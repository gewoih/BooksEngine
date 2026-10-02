using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Import;
using BooksEngine.Api.Recommendations;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class ImportEndpoints
{
    public static void MapImport(this RouteGroupBuilder api)
    {
        api.MapPost("/me/import", async (IFormFile file, ClaimsPrincipal user, NpgsqlDataSource ds, RecommenderClient svc,
            CancellationToken ct) =>
        {
            CsvFormat format;
            List<CsvRow> rows;
            try { (format, rows) = RatingsCsv.Parse(file.OpenReadStream()); }
            catch (FormatException e) { return Results.BadRequest(e.Message); }

            var errors = rows.Where(r => r.Error is not null).Select(r => new ImportIssue(r.Line, r.Label, r.Error!)).ToList();
            var skipped = rows.Count(r => r.Error is null && r.Rating == 0);           // Goodreads: 0 — без оценки
            var valid = rows.Where(r => r.Error is null && r.Rating > 0).ToList();
            var noId = valid.Where(r => r.ExternalId.Length == 0).ToList();
            var ids = valid.Where(r => r.ExternalId.Length > 0).Select(r => r.ExternalId).Distinct().ToArray();

            await using var c = await ds.OpenConnectionAsync(ct);
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
            var byId = (await c.QueryAsync<(string Ext, long WorkId, bool InCf)>(sql, new { ids }))
                .ToDictionary(f => f.Ext);
            var found = valid.Where(r => byId.ContainsKey(r.ExternalId))
                .ToDictionary(r => r, r => (byId[r.ExternalId].WorkId, byId[r.ExternalId].InCf));

            // без id — новая книга (после 2017)? Её ищет сервис выдачи по английскому названию и автору
            var why = "нет id";
            var byTitle = noId.Where(r => r.TitleEn is not null).ToList();
            if (byTitle.Count > 0)
                try
                {
                    var refs = await svc.MatchAsync(byTitle.Select(r => (r.TitleEn!, r.Author)).ToList(), ct);
                    var internalIds = await UserRatings.IdsAsync(c, refs.OfType<WorkRef>());
                    var inCf = (await c.QueryAsync<(long Id, bool InCf)>("SELECT id, in_cf FROM works WHERE id = ANY(@w)",
                        new { w = internalIds.Values.ToArray() })).ToDictionary(x => x.Id, x => x.InCf);
                    foreach (var (r, wr) in byTitle.Zip(refs))
                        if (wr is not null && internalIds.TryGetValue(wr, out var w)) found[r] = (w, inCf[w]);
                }
                catch (RecommenderUnavailableException) { why = "нет id (новые книги ищет сервис выдачи — он не запущен)"; }
            var notFound = noId.Where(r => !found.ContainsKey(r)).Select(r => new ImportIssue(r.Line, r.Label, why))
                .Concat(valid.Where(r => r.ExternalId.Length > 0 && !found.ContainsKey(r))
                    .Select(r => new ImportIssue(r.Line, r.Label, "нет в каталоге")))
                .ToList();

            // Несколько строк одного произведения (тень и главное) — средняя, округлённая как metrics.rounded;
            // «бросил» — если все строки книги dnf (как recommend.read_profile)
            var perWork = valid.Where(found.ContainsKey)
                .GroupBy(r => found[r].WorkId)
                .Select(g => (WorkId: g.Key, Value: (int)Math.Floor(g.Average(r => r.Rating!.Value) + 0.5),
                              Dnf: g.All(r => r.Dnf), First: g.First()))
                .ToList();
            var outside = perWork.Where(p => !found[p.First].InCf)
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
