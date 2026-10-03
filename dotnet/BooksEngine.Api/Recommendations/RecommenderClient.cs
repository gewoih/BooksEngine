using System.Net.Http.Json;
using System.Text.Json;

namespace BooksEngine.Api.Recommendations;

/// <summary>Книга для сервиса выдачи: источник (goodreads | amazon) и внешний id из external_ids.</summary>
public sealed record WorkRef(string Source, string Id);

public sealed record ServiceRating(string Source, string Id, int Rating, bool Dnf, string Title);
public sealed record ServiceItem(string Source, string Id, string Title, string Author, int Chance, List<string> Why)
{
    public WorkRef Ref => new(Source, Id);
}
public sealed record ServiceSection(string Name, string? Note, List<ServiceItem> Items);
public sealed record RankedRef(string Source, string Id, string Section)
{
    public WorkRef Ref => new(Source, Id);
}
/// <summary>Ranks — состав списков каждого размера из rank_tops (размер → книги по порядку, с названием списка).</summary>
public sealed record ServiceRecommendations(int Used, string ChanceLabel, string Legend, List<ServiceSection> Sections,
                                            Dictionary<string, List<RankedRef>> Ranks);
public sealed record ServiceChance(int? Chance, string ChanceLabel, bool InModel);
public sealed record ServiceRefs(List<WorkRef?> Items);

/// <summary>Сервис выдачи не запущен или не отвечает — эндпоинт выдачи отвечает 503.</summary>
public sealed class RecommenderUnavailableException(Exception inner)
    : Exception("сервис выдачи не отвечает: uv run booksengine --domain books-amazon serve", inner);

/// <summary>
/// Клиент `booksengine serve`: выдачу считает Python тем же кодом, что консольный `recommend`, — модель меняется часто,
/// и повтор её на C# отставал бы. Адрес — Recommender:Url (по умолчанию http://127.0.0.1:5090).
/// </summary>
public sealed class RecommenderClient(HttpClient http)
{
    public const string DefaultUrl = "http://127.0.0.1:5090";
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public Task<ServiceRecommendations> RecommendAsync(IReadOnlyList<ServiceRating> ratings, int top,
        IReadOnlyList<int> rankTops, CancellationToken ct) =>
        PostAsync<ServiceRecommendations>("recommend", new { ratings, top, rank_tops = rankTops }, ct);

    public Task<ServiceChance> ChanceAsync(IReadOnlyList<ServiceRating> ratings, WorkRef work, CancellationToken ct) =>
        PostAsync<ServiceChance>("chance", new { ratings, work }, ct);

    /// <summary>Новые книги (Amazon) по английскому названию и автору; не нашлась — null.</summary>
    public async Task<List<WorkRef?>> MatchAsync(IReadOnlyList<(string TitleEn, string? Author)> rows, CancellationToken ct) =>
        (await PostAsync<ServiceRefs>("match", new { rows = rows.Select(r => new { title_en = r.TitleEn, author = r.Author }) }, ct)).Items;

    public async Task<List<WorkRef>> SimilarAsync(WorkRef work, CancellationToken ct) =>
        (await SendAsync<ServiceRefs>(new HttpRequestMessage(HttpMethod.Get,
            $"similar?source={Uri.EscapeDataString(work.Source)}&id={Uri.EscapeDataString(work.Id)}"), ct)).Items
        .OfType<WorkRef>().ToList();

    private Task<T> PostAsync<T>(string path, object body, CancellationToken ct) =>
        // тело целиком, с Content-Length: http.server в Python не читает chunked
        SendAsync<T>(new HttpRequestMessage(HttpMethod.Post, path)
        {
            Content = new StringContent(JsonSerializer.Serialize(body, Json), System.Text.Encoding.UTF8, "application/json"),
        }, ct);

    private async Task<T> SendAsync<T>(HttpRequestMessage request, CancellationToken ct)
    {
        HttpResponseMessage r;
        try { r = await http.SendAsync(request, ct); }
        catch (HttpRequestException e) { throw new RecommenderUnavailableException(e); }
        catch (TaskCanceledException e) when (!ct.IsCancellationRequested) { throw new RecommenderUnavailableException(e); }
        using (r)
        {
            if (!r.IsSuccessStatusCode)
                throw new InvalidOperationException($"сервис выдачи: {(int)r.StatusCode} {await r.Content.ReadAsStringAsync(ct)}");
            return (await r.Content.ReadFromJsonAsync<T>(Json, ct))!;
        }
    }
}
