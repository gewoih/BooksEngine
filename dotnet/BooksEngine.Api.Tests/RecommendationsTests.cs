using System.Net;
using System.Net.Http.Json;

namespace BooksEngine.Api.Tests;

[Collection("db")]
public sealed class RecommendationsTests(TestDb db)
{
    private static async Task<RecommendationsDto> Recs(HttpClient c, int n = 10) =>
        (await c.GetFromJsonAsync<RecommendationsDto>($"/api/me/recommendations?n={n}"))!;

    [Fact]
    public async Task Sends_ratings_by_external_id_and_maps_sections_back_to_catalog()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("recs");
        await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(5, false));
        await c.PutAsJsonAsync("/api/me/ratings/1003", new RateRequest(null, true));

        var r = await Recs(c);
        var sent = api.Recommender.LastRatings!.AsArray();
        Assert.Equal(["1", "3"], sent.Select(x => x!["id"]!.GetValue<string>()));          // Goodreads-id, не внутренние
        Assert.True(sent[1]!["dnf"]!.GetValue<bool>());
        Assert.Equal("Дюна", sent[0]!["title"]!.GetValue<string>());                       // по-русски — для подписей

        Assert.Equal(2, r.UsedRatings);
        Assert.Equal("5★", r.ChanceLabel);
        Assert.Equal(["Художественная литература", "Новинки (после 2017)"], r.Sections.Select(s => s.Name));
        Assert.Equal([1006L, 1004L, 1002L], r.Sections[0].Items.Select(i => i.Book.WorkId));
        Assert.Equal([1, 2, 3], r.Sections[0].Items.Select(i => i.Rank));
        Assert.Equal(1008L, r.Sections[1].Items.Single().Book.WorkId);                        // новая книга — по ключу
        Assert.Equal("примерный шанс", r.Sections[1].Note);
        Assert.All(r.Sections.SelectMany(s => s.Items), i => Assert.Null(i.PreviousRank));    // сравнивать не с чем
        Assert.Empty(r.Changes);
    }

    [Fact]
    public async Task After_a_new_rating_shows_previous_places_and_what_changed()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("moves");
        await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(5, false));
        await Recs(c);                                                // Hobbit 1, Solaris 2, Emma 3, Messiah 4

        await c.PutAsJsonAsync("/api/me/ratings/1006", new RateRequest(4, false));
        var r = await Recs(c);                                        // Solaris 1, Emma 2, Messiah 3
        var byId = r.Sections[0].Items.ToDictionary(i => i.Book.WorkId);
        Assert.Equal((1, 2), (byId[1003].Rank, byId[1003].PreviousRank!.Value));
        Assert.Equal((3, 4), (byId[1002].Rank, byId[1002].PreviousRank!.Value));
        Assert.False(byId[1003].IsNew);
        Assert.Equal(["The Hobbit: оценка 4★"], r.Changes);

        var again = await Recs(c, 20);                                // те же оценки — сравнение с тем же прежним
        Assert.Equal(2, again.Sections[0].Items.Single(i => i.Book.WorkId == 1003).PreviousRank);

        await c.DeleteAsync("/api/me/ratings/1006");
        var back = await Recs(c);
        var hobbit = back.Sections[0].Items.Single(i => i.Book.WorkId == 1006);
        Assert.True(hobbit.IsNew);                                    // при оценённом Хоббите его в списке не было
        Assert.Equal(["The Hobbit: оценка 4★ удалена"], back.Changes);
    }

    [Fact]
    public async Task No_ratings_gives_empty_list_and_bad_n_is_rejected()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("empty");
        Assert.Empty((await Recs(c, 20)).Sections);
        Assert.Equal(HttpStatusCode.BadRequest, (await c.GetAsync("/api/me/recommendations?n=7")).StatusCode);
    }

    [Fact]
    public async Task Work_card_has_details_chance_and_similar()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("card");
        await c.PutAsJsonAsync("/api/me/ratings/1003", new RateRequest(5, false));
        var dune = (await c.GetFromJsonAsync<WorkDetailDto>("/api/works/1001"))!;
        Assert.Equal("spice", dune.Description);
        Assert.Equal(["Фрэнк Герберт"], dune.Authors);
        Assert.True(dune.InModel);
        Assert.Equal("https://images.gr-assets.com/books/1l/1.jpg", dune.LargeCoverUrl);
        Assert.Equal((42, "5★"), (dune.ChancePct, dune.ChanceLabel));
        Assert.Equal([1002L, 1006L], dune.Similar.Select(b => b.WorkId));
        Assert.Null((await c.GetFromJsonAsync<WorkDetailDto>("/api/works/1003"))!.ChancePct);   // оценена — без шанса
        var rare = (await c.GetFromJsonAsync<WorkDetailDto>("/api/works/1005"))!;
        Assert.False(rare.InModel);
        Assert.Equal(HttpStatusCode.NotFound, (await c.GetAsync("/api/works/424242")).StatusCode);
    }

    [Fact]
    public async Task Without_recommender_service_recommendations_are_503_but_card_and_library_work()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("down");
        await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(5, false));
        api.Recommender.Down = true;
        Assert.Equal(HttpStatusCode.ServiceUnavailable, (await c.GetAsync("/api/me/recommendations?n=10")).StatusCode);
        var card = (await c.GetFromJsonAsync<WorkDetailDto>("/api/works/1006"))!;
        Assert.Null(card.ChancePct);
        Assert.Empty(card.Similar);
        Assert.Equal(HttpStatusCode.OK, (await c.GetAsync("/api/library")).StatusCode);
    }
}
