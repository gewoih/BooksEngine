using System.Security.Claims;
using BooksEngine.Api.Auth;
using BooksEngine.Api.Books;
using BooksEngine.Api.Recommendations;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class RecommendationsEndpoints
{
    public static readonly int[] Sizes = [10, 20, 50];

    public static void MapRecommendations(this RouteGroupBuilder api)
    {
        api.MapGet("/me/recommendations", async (int? n, ClaimsPrincipal user, NpgsqlDataSource ds, ModelStore store) =>
        {
            var size = n ?? 20;
            if (!Sizes.Contains(size)) return Results.BadRequest("n — 10, 20 или 50");
            var m = store.Current;
            if (m is null) return Results.Problem("модель не выгружена: uv run booksengine export-model", statusCode: 503);
            await using var c = await ds.OpenConnectionAsync();
            var s = await UserScoring.ForUserAsync(c, m, user.UserId());
            var picks = Recommender.Top(m, s, size);
            var ids = picks.Select(p => p.WorkId)
                .Concat(picks.SelectMany(p => p.Because))
                .Concat(picks.Where(p => p.Despite is not null).Select(p => p.Despite!.Value))
                .Distinct().ToList();
            var books = (await BookQueries.ByIdsAsync(c, ids, user.UserId())).ToDictionary(b => b.WorkId);
            BookRef Ref(long id) => new(id, books[id].Title);
            var items = picks.Select(p => new RecommendationDto(books[p.WorkId], p.ChancePct,
                p.Because.Select(Ref).ToList(), p.Despite is { } d ? Ref(d) : null)).ToList();
            return Results.Ok(new RecommendationsDto(items, s.UsedRatings));
        }).RequireAuthorization().Produces<RecommendationsDto>();
    }
}
