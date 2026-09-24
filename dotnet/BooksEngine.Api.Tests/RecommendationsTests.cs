using System.Net;
using System.Net.Http.Json;

namespace BooksEngine.Api.Tests;

[Collection("db")]
public sealed class RecommendationsTests(TestDb db)
{
    [Fact]
    public async Task Recommends_unrated_books_without_series_continuation_with_chance_and_reason()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("recs");
        await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(5, false));

        var r = (await c.GetFromJsonAsync<RecommendationsDto>("/api/me/recommendations?n=10"))!;
        var ids = r.Items.Select(i => i.Book.WorkId).ToList();
        Assert.Equal(1, r.UsedRatings);
        Assert.DoesNotContain(1001L, ids);        // оценённая
        Assert.DoesNotContain(1002L, ids);        // продолжение серии (work_exclusions)
        Assert.Contains(1006L, ids);              // сосед Dune по EASE
        Assert.All(r.Items, i => Assert.InRange(i.ChancePct, 0, 100));
        Assert.Equal("Dune", r.Items.Single(i => i.Book.WorkId == 1006).Because.Single().Title);
    }

    [Fact]
    public async Task No_ratings_gives_empty_list_and_bad_n_is_rejected()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("empty");
        Assert.Empty((await c.GetFromJsonAsync<RecommendationsDto>("/api/me/recommendations?n=20"))!.Items);
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
        Assert.Equal(["Frank Herbert"], dune.Authors);
        Assert.True(dune.InModel);
        Assert.Equal("https://images.gr-assets.com/books/1l/1.jpg", dune.LargeCoverUrl);
        Assert.NotNull(dune.ChancePct);
        Assert.Equal([1002L, 1006L], dune.Similar.Select(b => b.WorkId));    // EASE-строка Dune по убыванию веса
        var rare = (await c.GetFromJsonAsync<WorkDetailDto>("/api/works/1005"))!;
        Assert.False(rare.InModel);                                         // вне модели — «мало данных»
        Assert.Null(rare.ChancePct);
        Assert.Equal(HttpStatusCode.NotFound, (await c.GetAsync("/api/works/424242")).StatusCode);
    }

    [Fact]
    public async Task Without_exported_model_recommendations_are_503_but_library_works()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await db.ClearModelAsync();
        try
        {
            await using var api = new ApiFactory(db);
            var c = await api.LoggedInClientAsync("nomodel");
            Assert.Equal(HttpStatusCode.ServiceUnavailable, (await c.GetAsync("/api/me/recommendations?n=10")).StatusCode);
            Assert.Null((await c.GetFromJsonAsync<WorkDetailDto>("/api/works/1001"))!.ChancePct);
            Assert.Equal(HttpStatusCode.OK, (await c.GetAsync("/api/library")).StatusCode);
        }
        finally { await db.SeedModelAsync(); }
    }

    [Fact]
    public async Task Reload_picks_up_new_export()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("reload");
        await db.ExecAsync("UPDATE model_meta SET fingerprint = 'test2'");
        try
        {
            var r = await (await c.PostAsync("/api/admin/reload-model", null)).Content.ReadFromJsonAsync<ReloadDto>();
            Assert.Equal("test2", r!.Fingerprint);
        }
        finally { await db.ExecAsync("UPDATE model_meta SET fingerprint = 'test'"); }
    }
}
