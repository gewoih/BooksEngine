using System.Net;
using System.Net.Http.Json;
using System.Text;

namespace BooksEngine.Api.Tests;

[Collection("db")]
public sealed class ImportTests(TestDb db)
{
    private static async Task<HttpResponseMessage> Upload(HttpClient c, string csv, bool bom = false)
    {
        var bytes = (bom ? Encoding.UTF8.GetPreamble() : []).Concat(Encoding.UTF8.GetBytes(csv)).ToArray();
        var form = new MultipartFormDataContent { { new ByteArrayContent(bytes), "file", "ratings.csv" } };
        return await c.PostAsync("/api/me/import", form);
    }

    [Fact]
    public async Task Our_format_maps_shadows_dnf_and_reports_problems()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("imp");
        await c.PutAsJsonAsync("/api/me/ratings/1001", new RateRequest(2, false));   // перезапишется
        await c.PutAsJsonAsync("/api/me/ratings/1006", new RateRequest(4, false));   // нет в файле — останется

        var r = await (await Upload(c, """
            title,rating,status,goodreads_work_id
            Дюна,5,read,1
            "Эмма, тень",4,read,777
            Солярис,,dnf,3
            Редкая,3,read,5
            Чужая,3,read,999999
            Новая,4,read,
            Кривая,7,read,4
            """)).Content.ReadFromJsonAsync<ImportResultDto>();

        Assert.Equal((3, 1), (r!.Added, r.Updated));                   // 1004 (тень 777), 1003, 1005 — новые; 1001 — обновлена
        Assert.Equal(["Чужая", "Новая"], r.NotFound.Select(i => i.Book));
        Assert.Equal(8, r.Errors.Single().Line);                       // «Кривая»: оценка 7
        Assert.Equal("Редкая", r.OutsideCore.Single().Book);
        var mine = (await c.GetFromJsonAsync<List<MyRatingDto>>("/api/me/ratings"))!.ToDictionary(x => x.Book.WorkId, x => x.Book);
        Assert.Equal(5, mine[1001].MyRating);
        Assert.Equal(4, mine[1004].MyRating);
        Assert.True(mine[1003].MyDnf);
        Assert.Equal(4, mine[1006].MyRating);
    }

    [Fact]
    public async Task Goodreads_export_with_bom_and_quoted_commas()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("gr");
        var r = await (await Upload(c, """
            Book Id,Title,Author,My Rating,Exclusive Shelf
            1,"Dune, Deluxe Edition",Frank Herbert,5,read
            2,"Солярис",Stanisław Lem,0,to-read
            5,Хоббит,Tolkien,4,read
            777,"Эмма, издание тени",Jane Austen,3,read
            31337,"Unknown, Book",Nobody,3,read
            """, bom: true)).Content.ReadFromJsonAsync<ImportResultDto>();
        Assert.Equal((3, 0, 1), (r!.Added, r.Updated, r.Skipped));     // 0 — без оценки; издание 2005 → Хоббит
        Assert.Equal("Unknown, Book", r.NotFound.Single().Book);
        Assert.Empty(r.OutsideCore);                                   // издание тени 2777 → главное 1004, в ядре
        var mine = (await c.GetFromJsonAsync<List<MyRatingDto>>("/api/me/ratings"))!.ToDictionary(x => x.Book.WorkId, x => x.Book);
        Assert.Equal(3, mine[1004].MyRating);
        Assert.DoesNotContain(1777L, mine.Keys);
    }

    [Fact]
    public async Task Unknown_header_is_rejected()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен");
        await using var api = new ApiFactory(db);
        var c = await api.LoggedInClientAsync("hdr");
        Assert.Equal(HttpStatusCode.BadRequest, (await Upload(c, "a,b\n1,2\n")).StatusCode);
    }
}
