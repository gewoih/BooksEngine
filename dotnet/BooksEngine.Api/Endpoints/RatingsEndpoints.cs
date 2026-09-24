using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Books;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class RatingsEndpoints
{
    public static void MapRatings(this RouteGroupBuilder api)
    {
        var me = api.MapGroup("/me/ratings").RequireAuthorization();

        me.MapGet("/", async (ClaimsPrincipal user, NpgsqlDataSource ds) =>
        {
            await using var c = await ds.OpenConnectionAsync();
            var rows = (await c.QueryAsync<(long WorkId, DateTime UpdatedAt)>(
                "SELECT work_id, updated_at FROM ratings WHERE user_id = @u ORDER BY updated_at DESC, work_id",
                new { u = user.UserId() })).ToList();
            var books = (await BookQueries.ByIdsAsync(c, rows.Select(r => r.WorkId).ToList(), user.UserId()))
                .ToDictionary(b => b.WorkId);
            return rows.Select(r => new MyRatingDto(books[r.WorkId], r.UpdatedAt)).ToList();
        });

        me.MapPut("/{workId:long}", async (long workId, RateRequest req, ClaimsPrincipal user, NpgsqlDataSource ds) =>
        {
            var value = req.Dnf ? 1 : req.Value ?? 0;
            if (value is < 1 or > 5) return Results.BadRequest("оценка — целое 1–5 или «бросил»");
            await using var c = await ds.OpenConnectionAsync();
            if (!await c.ExecuteScalarAsync<bool>("SELECT EXISTS (SELECT 1 FROM works WHERE id = @workId)", new { workId }))
                return Results.NotFound("нет такой книги");
            await using var tx = await c.BeginTransactionAsync();
            await UpsertAsync(c, tx, user.UserId(), workId, value, req.Dnf);
            await tx.CommitAsync();
            return Results.Ok((await BookQueries.ByIdsAsync(c, [workId], user.UserId())).Single());
        }).Produces<BookDto>();

        me.MapDelete("/{workId:long}", async (long workId, ClaimsPrincipal user, NpgsqlDataSource ds) =>
        {
            await using var c = await ds.OpenConnectionAsync();
            await c.ExecuteAsync("""
                DELETE FROM ratings WHERE user_id = @u AND work_id = @workId;
                DELETE FROM shelves WHERE user_id = @u AND work_id = @workId AND status = 'dnf';
                """, new { u = user.UserId(), workId });
            return Results.NoContent();
        });
    }

    /// <summary>Оценка + полка dnf. Вызывается и из импорта — одна транзакция на файл.</summary>
    public static Task UpsertAsync(NpgsqlConnection c, NpgsqlTransaction tx, long userId, long workId, int value, bool dnf) =>
        c.ExecuteAsync(dnf
            ? """
              INSERT INTO ratings (user_id, work_id, value) VALUES (@userId, @workId, @value)
                ON CONFLICT (user_id, work_id) DO UPDATE SET value = excluded.value, updated_at = now();
              INSERT INTO shelves (user_id, work_id, status) VALUES (@userId, @workId, 'dnf')
                ON CONFLICT (user_id, work_id) DO UPDATE SET status = 'dnf', updated_at = now();
              """
            : """
              INSERT INTO ratings (user_id, work_id, value) VALUES (@userId, @workId, @value)
                ON CONFLICT (user_id, work_id) DO UPDATE SET value = excluded.value, updated_at = now();
              DELETE FROM shelves WHERE user_id = @userId AND work_id = @workId AND status = 'dnf';
              """, new { userId, workId, value = (short)value }, tx);
}
