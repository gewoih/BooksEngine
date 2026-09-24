using System.Text.RegularExpressions;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Books;

public static partial class BookQueries
{
    /// <summary>Книги по id в заданном порядке, с оценкой пользователя. Основной автор — без роли, первый по позиции.</summary>
    public static async Task<List<BookDto>> ByIdsAsync(NpgsqlConnection c, IReadOnlyList<long> ids, long userId)
    {
        if (ids.Count == 0) return [];
        var rows = await c.QueryAsync<BookDto>("""
            SELECT w.id AS work_id, w.title, a.name AS author, w.publication_year AS year, cv.image_url AS cover_url,
                   w.cf_ratings, r.value::int AS my_rating, coalesce(s.status = 'dnf', false) AS my_dnf, w.in_cf AS in_core
            FROM works w
            LEFT JOIN LATERAL (SELECT au.name FROM work_authors wa JOIN authors au ON au.id = wa.author_id
                               WHERE wa.work_id = w.id AND coalesce(wa.role, '') = ''
                               ORDER BY wa.position LIMIT 1) a ON true
            LEFT JOIN work_covers cv ON cv.work_id = w.id
            LEFT JOIN ratings r ON r.work_id = w.id AND r.user_id = @userId
            LEFT JOIN shelves s ON s.work_id = w.id AND s.user_id = @userId
            WHERE w.id = ANY(@ids)
            """, new { ids = ids.ToArray(), userId });
        var byId = rows.ToDictionary(b => b.WorkId);
        return ids.Where(byId.ContainsKey).Select(id => byId[id]).ToList();
    }

    [GeneratedRegex(@"(\d)m/")]
    private static partial Regex MediumSize();

    /// <summary>У обложек Goodreads размер — буква после числа: m — сетка, l — карточка.</summary>
    public static string? LargeCover(string? url) => url is null ? null : MediumSize().Replace(url, "$1l/", 1);
}
