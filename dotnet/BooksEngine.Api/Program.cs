using BooksEngine.Api.Endpoints;
using BooksEngine.Api.Recommendations;
using System.Text.Json.Serialization;
using BooksEngine.Db;
using Dapper;
using Microsoft.AspNetCore.Authentication.Cookies;

var builder = WebApplication.CreateBuilder(args);
var connectionString = builder.Configuration.GetConnectionString("Books") ?? ConnectionString.FromEnvironment();

DefaultTypeMap.MatchNamesWithUnderscores = true;   // snake_case колонки → свойства DTO
builder.Services.AddNpgsqlDataSource(connectionString);
// числа — только числами: иначе OpenAPI описывает их как number | string и типы фронта размыты
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.NumberHandling = JsonNumberHandling.Strict);
builder.Services.AddOpenApi();
builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(o =>
    {
        o.Cookie.Name = "booksengine_session";
        o.ExpireTimeSpan = TimeSpan.FromDays(365);
        // API, а не сайт со страницей входа: без сессии — 401, а не редирект
        o.Events.OnRedirectToLogin = ctx => { ctx.Response.StatusCode = 401; return Task.CompletedTask; };
    });
builder.Services.AddAuthorization();
builder.Services.AddSingleton<ModelStore>();

var app = builder.Build();
app.UseAuthentication();
app.UseAuthorization();
app.MapOpenApi();

var api = app.MapGroup("/api");
api.MapUsers();
api.MapLibrary();
api.MapRatings();
api.MapRecommendations();
api.MapWorks();
api.MapAdmin();
api.MapImport();

// Модель не читается (например, load-db каскадом удалил её строки) — API всё равно стартует: библиотека и оценки
// работают, выдача отвечает 503 до export-model и reload
try { await app.Services.GetRequiredService<ModelStore>().ReloadAsync(CancellationToken.None); }
catch (Exception e) { app.Logger.LogError(e, "Модель не загружена: uv run booksengine export-model, затем reload-model"); }
await app.RunAsync();

public partial class Program;   // для WebApplicationFactory в тестах
