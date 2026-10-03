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
        api.MapGet("/me/recommendations", async (int? n, ClaimsPrincipal user, NpgsqlDataSource ds, RecommenderClient svc,
            CancellationToken ct) =>
        {
            var size = n ?? 20;
            if (!Sizes.Contains(size)) return Results.BadRequest("n — 10, 20 или 50");
            await using var c = await ds.OpenConnectionAsync(ct);
            var userId = user.UserId();
            var ratings = await UserRatings.LoadAsync(c, userId);
            if (ratings.Count == 0) return Results.Ok(new RecommendationsDto([], 0, "", "", []));

            ServiceRecommendations recs;
            try { recs = await svc.RecommendAsync(UserRatings.ForService(ratings), size, Sizes, ct); }
            catch (RecommenderUnavailableException e) { return Results.Problem(e.Message, statusCode: 503); }

            // места для стрелок — в своём списке, у каждого размера списка (правила списка зависят от размера)
            var ids = await UserRatings.IdsAsync(c, recs.Sections.SelectMany(s => s.Items).Select(i => i.Ref)
                .Concat(recs.Ranks.Values.SelectMany(v => v).Select(r => r.Ref)));
            var ranks = recs.Ranks.ToDictionary(kv => int.Parse(kv.Key), kv => kv.Value.Where(r => ids.ContainsKey(r.Ref))
                .GroupBy(r => r.Section).SelectMany(g => g.Select((r, k) => (Id: ids[r.Ref], Rank: k + 1)))
                .ToDictionary(t => t.Id, t => t.Rank));
            var cmp = await RankChanges.SaveAndCompareAsync(c, userId,
                ratings.ToDictionary(r => r.WorkId, RankChanges.Label), ranks);
            var before = cmp?.Before.GetValueOrDefault(size);

            var shown = recs.Sections.Select(s => (s.Name, s.Note,
                Items: s.Items.Where(i => ids.ContainsKey(i.Ref)).Select((i, k) => (Id: ids[i.Ref], Rank: k + 1, i)).ToList()))
                .ToList();
            var changed = cmp?.Changes ?? [];
            var books = (await BookQueries.ByIdsAsync(c,
                    shown.SelectMany(s => s.Items).Select(i => i.Id).Concat(changed.Select(ch => ch.WorkId)).Distinct().ToList(),
                    userId))
                .ToDictionary(b => b.WorkId);
            var result = shown.Select(s => new RecommendationSectionDto(s.Name, s.Note, s.Items
                .Where(i => books.ContainsKey(i.Id))
                .Select(i => new RecommendationDto(books[i.Id], i.Rank, i.i.Chance, i.i.Why,
                    before?.TryGetValue(i.Id, out var was) == true ? was : null,
                    before is not null && !before.ContainsKey(i.Id)))
                .ToList())).ToList();
            var changes = changed.Where(ch => books.ContainsKey(ch.WorkId))
                .Select(ch => $"{books[ch.WorkId].Title}: {Describe(ch)}").ToList();
            return Results.Ok(new RecommendationsDto(result, recs.Used, recs.ChanceLabel, recs.Legend, changes));
        }).RequireAuthorization().Produces<RecommendationsDto>();
    }

    private static string Stars(string label) => label == "dnf" ? "не дочитал" : $"{label}★";

    private static string Describe(RatingChange ch) => (ch.Before, ch.After) switch
    {
        (null, { } a) => $"оценка {Stars(a)}",
        ({ } b, null) => $"оценка {Stars(b)} удалена",
        ({ } b, { } a) => $"{Stars(b)} → {Stars(a)}",
        _ => "",
    };
}
