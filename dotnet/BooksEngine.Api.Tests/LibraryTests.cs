using System.Net;
using System.Net.Http.Json;

namespace BooksEngine.Api.Tests;

[Collection("db")]
public sealed class LibraryTests(TestDb db)
{
    private static async Task<PageDto<BookDto>> Get(HttpClient c, string query) =>
        (await c.GetFromJsonAsync<PageDto<BookDto>>("/api/library" + query))!;

    [Fact]
    public async Task Without_query_shows_core_by_popularity_with_cover_and_author()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("lib");
        var page = await Get(c, "");
        Assert.Equal(5, page.Total);                                               // Rare Book вне ядра — нет
        Assert.Equal(["The Hobbit", "Dune", "Emma", "Dune Messiah", "Solaris"], page.Items.Select(b => b.Title));
        var dune = page.Items.Single(b => b.WorkId == 1001);
        Assert.Equal(("Frank Herbert", (int?)1965, "https://images.gr-assets.com/books/1m/1.jpg", true),
                     (dune.Author, dune.Year, dune.CoverUrl, dune.InCore));
    }

    [Theory]
    [InlineData("?q=dune", new long[] { 1001, 1002 })]          // точное название — первым
    [InlineData("?q=солярис", new long[] { 1003 })]             // по названию издания (русские — только там)
    [InlineData("?q=austen", new long[] { 1004 })]              // по автору; Rare Book того же автора — вне ядра
    [InlineData("?q=du", new long[] { 1001, 1002 })]            // 1–2 символа: по началу названия
    public async Task Search_finds_by_title_edition_author_and_prefix(string query, long[] expected)
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("search");
        Assert.Equal(expected, (await Get(c, query)).Items.Select(b => b.WorkId));
    }

    [Fact]
    public async Task Library_requires_session()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        Assert.Equal(HttpStatusCode.Unauthorized, (await api.CreateClient().GetAsync("/api/library")).StatusCode);
    }

    [Fact]
    public void Large_cover_swaps_size_letter()
    {
        Assert.Equal("https://images.gr-assets.com/books/1451751461l/28431755.jpg",
            Books.BookQueries.LargeCover("https://images.gr-assets.com/books/1451751461m/28431755.jpg"));
        Assert.Null(Books.BookQueries.LargeCover(null));
    }
}
