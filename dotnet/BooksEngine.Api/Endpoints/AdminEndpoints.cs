using BooksEngine.Api.Recommendations;

namespace BooksEngine.Api.Endpoints;

public static class AdminEndpoints
{
    // Без отдельной роли: приложение локальное; при выкладке — закрыть вместе с настоящей аутентификацией.
    public static void MapAdmin(this RouteGroupBuilder api) =>
        api.MapPost("/admin/reload-model", async (ModelStore store, CancellationToken ct) =>
            new ReloadDto(await store.ReloadAsync(ct))).RequireAuthorization();
}
