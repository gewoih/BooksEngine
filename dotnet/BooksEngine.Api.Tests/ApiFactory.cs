using System.Net.Http.Json;
using BooksEngine.Api.Recommendations;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;

namespace BooksEngine.Api.Tests;

public sealed class ApiFactory(TestDb db) : WebApplicationFactory<Program>
{
    /// <summary>Сервис выдачи — подделка (Python в тестах API не запускается).</summary>
    public FakeRecommender Recommender { get; } = new();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseSetting("ConnectionStrings:Books", db.ConnectionString);
        builder.ConfigureTestServices(s => s.AddHttpClient<RecommenderClient>()
            .ConfigurePrimaryHttpMessageHandler(() => Recommender));
    }

    /// <summary>Клиент с сессией нового пользователя (уникальное имя — тесты не мешают друг другу).</summary>
    public async Task<HttpClient> LoggedInClientAsync(string name)
    {
        var client = CreateClient();
        var created = await client.PostAsJsonAsync("/api/users", new CreateUserRequest($"{name}-{Guid.NewGuid():N}"));
        created.EnsureSuccessStatusCode();
        var user = (await created.Content.ReadFromJsonAsync<UserDto>())!;
        (await client.PostAsJsonAsync("/api/session", new SessionRequest(user.Id))).EnsureSuccessStatusCode();
        return client;
    }
}
