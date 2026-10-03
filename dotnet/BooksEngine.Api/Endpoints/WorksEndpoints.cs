using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Books;
using BooksEngine.Api.Recommendations;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class WorksEndpoints
{
    public static void MapWorks(this RouteGroupBuilder api)
    {
        api.MapGet("/works/{id:long}", async (long id, ClaimsPrincipal user, NpgsqlDataSource ds, RecommenderClient svc,
            CancellationToken ct) =>
        {
            await using var c = await ds.OpenConnectionAsync(ct);
            var userId = user.UserId();
            var book = (await BookQueries.ByIdsAsync(c, [id], userId)).SingleOrDefault();
            if (book is null) return Results.NotFound("нет такой книги");
            var description = await c.ExecuteScalarAsync<string?>("SELECT description FROM works WHERE id = @id", new { id });
            var authors = (await c.QueryAsync<string>("""
                SELECT coalesce(a.ru_name, a.name) FROM work_authors wa JOIN authors a ON a.id = wa.author_id
                WHERE wa.work_id = @id AND coalesce(wa.role, '') = '' AND a.name IS NOT NULL ORDER BY wa.position
                """, new { id })).ToList();
            var genres = (await c.QueryAsync<string>("""
                SELECT g.name FROM work_genres wg JOIN genres g ON g.id = wg.genre_id WHERE wg.work_id = @id
                ORDER BY wg.votes DESC, g.name
                """, new { id })).ToList();

            // шанс и похожие — от сервиса выдачи; он не запущен — карточка без них
            ServiceChance? chance = null;
            var similar = new List<BookDto>();
            if (await UserRatings.RefAsync(c, id) is { } work)
                try
                {
                    var ratings = UserRatings.ForService(await UserRatings.LoadAsync(c, userId));
                    chance = await svc.ChanceAsync(ratings, work, ct);
                    var refs = await svc.SimilarAsync(work, ct);
                    var ids = await UserRatings.IdsAsync(c, refs);
                    similar = await BookQueries.ByIdsAsync(c, refs.Where(ids.ContainsKey).Select(r => ids[r]).ToList(), userId);
                }
                catch (RecommenderUnavailableException) { }
            return Results.Ok(new WorkDetailDto(book, BookQueries.LargeCover(book.CoverUrl), description, authors, genres,
                book.MyRating is null ? chance?.Chance : null, chance?.ChanceLabel, chance?.InModel ?? false, similar));
        }).RequireAuthorization().Produces<WorkDetailDto>();
    }
}
