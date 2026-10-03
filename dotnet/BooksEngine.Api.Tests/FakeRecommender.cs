using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace BooksEngine.Api.Tests;

/// <summary>
/// Подделка `booksengine serve`: выдачу считает Python, здесь проверяется только, что API передаёт ему и как
/// раскладывает ответ. Советует по порядку Order всё неоценённое (Goodreads-id; «amazon» — новая книга), шанс 42,
/// похожие на Dune — Dune Messiah и Хоббит, новая книга находится по названию «New Book». Down — сервис не запущен.
/// </summary>
public sealed class FakeRecommender : HttpMessageHandler
{
    public static readonly string[] Order = ["6", "3", "4", "2", "1"];
    public const string NewKey = "author|new book";
    public bool Down { get; set; }
    public JsonNode? LastRatings { get; private set; }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        if (Down) throw new HttpRequestException("connection refused");
        var path = request.RequestUri!.AbsolutePath.Trim('/');
        // настоящий сервис (http.server) chunked не читает — тело должно идти с длиной
        if (request.Content is not null && request.Content.Headers.ContentLength is null)
            throw new InvalidOperationException("тело без Content-Length");
        var body = request.Content is null ? null : JsonNode.Parse(await request.Content.ReadAsStringAsync(ct));
        object answer = path switch
        {
            "recommend" => Recommend(body!),
            "chance" => new { chance = 42, chance_label = "5★", in_model = body!["work"]!["id"]!.GetValue<string>() != "5" },
            "similar" => new { items = request.RequestUri.Query.Contains("id=1&") || request.RequestUri.Query.EndsWith("id=1")
                ? new[] { new { source = "goodreads", id = "2" }, new { source = "goodreads", id = "6" } } : [] },
            "match" => new { items = body!["rows"]!.AsArray().Select(r => r!["title_en"]!.GetValue<string>() == "New Book"
                ? new { source = "amazon", id = NewKey } : null) },
            _ => throw new InvalidOperationException(path),
        };
        return new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(JsonSerializer.Serialize(answer), Encoding.UTF8, "application/json"),
        };
    }

    private object Recommend(JsonNode body)
    {
        LastRatings = body["ratings"];
        var rated = body["ratings"]!.AsArray().Select(r => r!["id"]!.GetValue<string>()).ToHashSet();
        var items = Order.Where(id => !rated.Contains(id)).Select(id => new
        {
            source = "goodreads", id, title = $"T{id}", author = "A", chance = 42, why = new[] { $"читатели: {rated.First()}" },
        });
        var fresh = rated.Contains(NewKey) ? [] : new[]
        {
            new { source = "amazon", id = NewKey, title = "New", author = "A", chance = 20, why = new[] { "по профилю в целом" } },
        };
        var tops = body["rank_tops"]!.AsArray().Select(k => k!.GetValue<int>());
        var ranks = tops.ToDictionary(k => k.ToString(), k => items.Take(k)
            .Select(i => new { i.source, i.id, section = "Художественная литература" })
            .Concat(fresh.Select(i => new { i.source, i.id, section = "Новинки (после 2017)" })).ToList());
        var top = body["top"]!.GetValue<int>();
        return new
        {
            used = rated.Count, chance_label = "5★", legend = "как читать", ranks,
            sections = new object[]
            {
                new { name = "Художественная литература", note = (string?)null, items = items.Take(top) },
                new { name = "Новинки (после 2017)", note = "примерный шанс", items = fresh },
            },
        };
    }
}
