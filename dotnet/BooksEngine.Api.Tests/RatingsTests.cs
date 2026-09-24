using System.Net;
using System.Net.Http.Json;

namespace BooksEngine.Api.Tests;

[Collection("db")]
public sealed class RatingsTests(TestDb db)
{
    private static async Task<Dictionary<long, BookDto>> Mine(HttpClient c) =>
        (await c.GetFromJsonAsync<List<MyRatingDto>>("/api/me/ratings"))!.ToDictionary(r => r.Book.WorkId, r => r.Book);

    [Fact]
    public async Task Rate_change_dnf_and_delete()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("rate");

        var dune = await (await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(5, false))).Content.ReadFromJsonAsync<BookDto>();
        Assert.Equal((5, false), (dune!.MyRating, dune.MyDnf));
        await c.PutAsJsonAsync("/api/me/ratings/1003", new RateRequest(null, true));       // бросил = 1 + dnf
        await c.PutAsJsonAsync("/api/me/ratings/1005", new RateRequest(4, false));         // вне ядра — можно

        var list = (await c.GetFromJsonAsync<List<MyRatingDto>>("/api/me/ratings"))!;
        Assert.Equal([1005L, 1003, 1001], list.Select(r => r.Book.WorkId));                 // последние сверху
        Assert.Equal((1, true), (list[1].Book.MyRating, list[1].Book.MyDnf));
        Assert.False(list[0].Book.InCore);

        await c.PutAsJsonAsync("/api/me/ratings/1003", new RateRequest(3, false));          // передумал: dnf снимается
        var solaris = (await Mine(c))[1003];
        Assert.Equal((3, false), (solaris.MyRating, solaris.MyDnf));

        Assert.Equal(HttpStatusCode.NoContent, (await c.DeleteAsync("/api/me/ratings/1001")).StatusCode);
        Assert.DoesNotContain(1001L, (await Mine(c)).Keys);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(6)]
    public async Task Rating_outside_one_to_five_is_rejected(int value)
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("bad");
        Assert.Equal(HttpStatusCode.BadRequest, (await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(value, false))).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await c.PutAsJsonAsync("/api/me/ratings/424242", new RateRequest(3, false))).StatusCode);
    }
}
