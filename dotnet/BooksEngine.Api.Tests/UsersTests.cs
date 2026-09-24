using System.Net;
using System.Net.Http.Json;

namespace BooksEngine.Api.Tests;

[Collection("db")]
public sealed class UsersTests(TestDb db)
{
    [Fact]
    public async Task Created_user_can_log_in_and_session_returns_him()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен (docker compose up -d)");
        await using var api = new ApiFactory(db);
        var client = api.CreateClient();
        Assert.Equal(HttpStatusCode.Unauthorized, (await client.GetAsync("/api/session")).StatusCode);

        var user = await (await client.PostAsJsonAsync("/api/users", new CreateUserRequest("Никита-" + Guid.NewGuid())))
            .Content.ReadFromJsonAsync<UserDto>();
        (await client.PostAsJsonAsync("/api/session", new SessionRequest(user!.Id))).EnsureSuccessStatusCode();

        Assert.Equal(user, await client.GetFromJsonAsync<UserDto>("/api/session"));
        Assert.Contains(user, (await client.GetFromJsonAsync<List<UserDto>>("/api/users"))!);
    }

    [Fact]
    public async Task Duplicate_or_empty_name_is_rejected_and_unknown_user_cannot_log_in()
    {
        Assert.SkipUnless(db.Available, "PostgreSQL недоступен (docker compose up -d)");
        await using var api = new ApiFactory(db);
        var client = api.CreateClient();
        var name = "Дубль-" + Guid.NewGuid();
        (await client.PostAsJsonAsync("/api/users", new CreateUserRequest(name))).EnsureSuccessStatusCode();
        Assert.Equal(HttpStatusCode.Conflict, (await client.PostAsJsonAsync("/api/users", new CreateUserRequest(name))).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await client.PostAsJsonAsync("/api/users", new CreateUserRequest("  "))).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.PostAsJsonAsync("/api/session", new SessionRequest(-1))).StatusCode);
    }
}
