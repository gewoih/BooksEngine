using System.Security.Claims;
using BooksEngine.Api.Auth;
using Dapper;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Npgsql;

namespace BooksEngine.Api.Endpoints;

public static class UsersEndpoints
{
    public static void MapUsers(this RouteGroupBuilder api)
    {
        api.MapGet("/users", async (NpgsqlDataSource ds) =>
        {
            await using var c = await ds.OpenConnectionAsync();
            return (await c.QueryAsync<UserDto>("SELECT id, name FROM app_users ORDER BY name")).ToList();
        });

        api.MapPost("/users", async (CreateUserRequest req, NpgsqlDataSource ds) =>
        {
            var name = req.Name?.Trim() ?? "";
            if (name.Length == 0) return Results.BadRequest("имя пустое");
            await using var c = await ds.OpenConnectionAsync();
            var user = await c.QuerySingleOrDefaultAsync<UserDto>(
                "INSERT INTO app_users (name) VALUES (@name) ON CONFLICT (name) DO NOTHING RETURNING id, name", new { name });
            return user is null ? Results.Conflict("такое имя уже есть") : Results.Ok(user);
        }).Produces<UserDto>();

        api.MapGet("/session", async (ClaimsPrincipal principal, NpgsqlDataSource ds) =>
        {
            if (principal.Identity?.IsAuthenticated != true) return Results.Unauthorized();
            await using var c = await ds.OpenConnectionAsync();
            var user = await c.QuerySingleOrDefaultAsync<UserDto>("SELECT id, name FROM app_users WHERE id = @id",
                new { id = principal.UserId() });
            return user is null ? Results.Unauthorized() : Results.Ok(user);
        }).Produces<UserDto>();

        api.MapPost("/session", async (SessionRequest req, HttpContext http, NpgsqlDataSource ds) =>
        {
            await using var c = await ds.OpenConnectionAsync();
            var user = await c.QuerySingleOrDefaultAsync<UserDto>("SELECT id, name FROM app_users WHERE id = @id",
                new { id = req.UserId });
            if (user is null) return Results.NotFound("нет такого пользователя");
            var identity = new ClaimsIdentity([new Claim(ClaimTypes.NameIdentifier, user.Id.ToString())],
                CookieAuthenticationDefaults.AuthenticationScheme);
            await http.SignInAsync(new ClaimsPrincipal(identity));
            return Results.Ok(user);
        }).Produces<UserDto>();

        api.MapDelete("/session", async (HttpContext http) =>
        {
            await http.SignOutAsync();
            return Results.NoContent();
        });
    }
}
