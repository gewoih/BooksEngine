using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Books;
using BooksEngine.Api.Recommendations;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class WorksEndpoints
{
    private const int SimilarCount = 10;

    public static void MapWorks(this RouteGroupBuilder api)
    {
        api.MapGet("/works/{id:long}", async (long id, ClaimsPrincipal user, NpgsqlDataSource ds, ModelStore store) =>
        {
            await using var c = await ds.OpenConnectionAsync();
            var book = (await BookQueries.ByIdsAsync(c, [id], user.UserId())).SingleOrDefault();
            if (book is null) return Results.NotFound("нет такой книги");
            var description = await c.ExecuteScalarAsync<string?>("SELECT description FROM works WHERE id = @id", new { id });
            var authors = (await c.QueryAsync<string>("""
                SELECT a.name FROM work_authors wa JOIN authors a ON a.id = wa.author_id
                WHERE wa.work_id = @id AND coalesce(wa.role, '') = '' AND a.name IS NOT NULL ORDER BY wa.position
                """, new { id })).ToList();
            var genres = (await c.QueryAsync<string>("""
                SELECT g.name FROM work_genres wg JOIN genres g ON g.id = wg.genre_id WHERE wg.work_id = @id
                ORDER BY wg.votes DESC, g.name
                """, new { id })).ToList();

            // шанс и похожие — только у книг EASE (30 000): у остальных «мало данных»
            var m = store.Current;
            int? chance = null;
            var similar = new List<BookDto>();
            if (m is not null && m.InEase(id))
            {
                if (book.MyRating is null)
                    chance = Recommender.ChanceFor(m, await UserScoring.ForUserAsync(c, m, user.UserId()), id);
                var pos = m.EasePosByCol[m.ColByWork[id]];
                var ids = Enumerable.Range(m.EaseRowPtr[pos], m.EaseRowPtr[pos + 1] - m.EaseRowPtr[pos])
                    .Where(e => m.EaseWeight[e] > 0).OrderByDescending(e => m.EaseWeight[e]).Take(SimilarCount)
                    .Select(e => m.WorkIdByCol[m.EaseCols[m.EaseTo[e]]]).ToList();
                similar = await BookQueries.ByIdsAsync(c, ids, user.UserId());
            }
            return Results.Ok(new WorkDetailDto(book, BookQueries.LargeCover(book.CoverUrl), description, authors, genres,
                chance, m?.InEase(id) ?? false, similar));
        }).RequireAuthorization().Produces<WorkDetailDto>();
    }
}
