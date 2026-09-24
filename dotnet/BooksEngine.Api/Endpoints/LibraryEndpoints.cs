using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Books;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class LibraryEndpoints
{
    public const int PageSize = 50;

    // Поиск (docs/resheniya.md, «поиск по каталогу»): сходство по целым словам (strict_word_similarity, <<%) —
    // word_similarity (<%) цеплял куски слов: «king» находил «thinking» и ставил «Гарри Поттера» первым; названия
    // изданий сворачиваются в произведение (русские названия есть только там); автор. Порядок — точное название,
    // сходство (до сотых), популярность. 1–2 символа: у триграмм нет совпадений — ищем по началу названия.
    private const string SearchSql = """
        WITH hits AS (
            SELECT w.id AS work_id, lower(w.title) = lower(@q) AS exact, strict_word_similarity(@q, w.title) AS sim
            FROM works w WHERE length(@q) >= 3 AND @q <<% w.title
            UNION ALL
            SELECT e.work_id, lower(e.title) = lower(@q), strict_word_similarity(@q, e.title)
            FROM editions e WHERE length(@q) >= 3 AND @q <<% e.title
            UNION ALL
            SELECT wa.work_id, false, strict_word_similarity(@q, a.name)
            FROM authors a JOIN work_authors wa ON wa.author_id = a.id WHERE length(@q) >= 3 AND @q <<% a.name
            UNION ALL
            SELECT w.id, lower(w.title) = lower(@q), 1.0 FROM works w
            WHERE length(@q) < 3 AND w.title ILIKE @prefix
        ),
        found AS (SELECT work_id, bool_or(exact) AS exact, max(sim) AS sim FROM hits GROUP BY work_id)
        SELECT f.work_id FROM found f JOIN works w ON w.id = f.work_id
        WHERE w.in_cf
        ORDER BY f.exact DESC, round(f.sim::numeric, 2) DESC, w.cf_ratings DESC, w.id
        """;

    public static void MapLibrary(this RouteGroupBuilder api)
    {
        api.MapGet("/library", async (string? q, int? page, ClaimsPrincipal user, NpgsqlDataSource ds) =>
        {
            var p = Math.Max(page ?? 1, 1);
            await using var c = await ds.OpenConnectionAsync();
            if (string.IsNullOrWhiteSpace(q))
            {
                var total = await c.ExecuteScalarAsync<int>("SELECT count(*) FROM works WHERE in_cf");
                var ids = (await c.QueryAsync<long>(
                    "SELECT id FROM works WHERE in_cf ORDER BY cf_ratings DESC, id LIMIT @n OFFSET @off",
                    new { n = PageSize, off = (p - 1) * PageSize })).ToList();
                return new PageDto<BookDto>(await BookQueries.ByIdsAsync(c, ids, user.UserId()), p, PageSize, total);
            }
            var query = q.Trim();
            var all = (await c.QueryAsync<long>(SearchSql, new { q = query, prefix = EscapeLike(query) + "%" })).ToList();
            var pageIds = all.Skip((p - 1) * PageSize).Take(PageSize).ToList();
            return new PageDto<BookDto>(await BookQueries.ByIdsAsync(c, pageIds, user.UserId()), p, PageSize, all.Count);
        }).RequireAuthorization();
    }

    private static string EscapeLike(string s) => s.Replace(@"\", @"\\").Replace("%", @"\%").Replace("_", @"\_");
}
