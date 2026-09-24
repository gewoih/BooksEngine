using System.Security.Claims;

namespace BooksEngine.Api.Auth;

/// <summary>
/// Пользователь запроса. Сейчас сессия — cookie «войти как» (без пароля, приложение локальное);
/// настоящая аутентификация заменит только выдачу cookie, эндпоинты берут id отсюда.
/// </summary>
public static class CurrentUser
{
    public static long UserId(this ClaimsPrincipal user) =>
        long.Parse(user.FindFirstValue(ClaimTypes.NameIdentifier)
                   ?? throw new InvalidOperationException("эндпоинт без RequireAuthorization"));
}
