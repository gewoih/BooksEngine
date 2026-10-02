using Dapper;
using Npgsql;

namespace BooksEngine.Api.Recommendations;

public sealed record UserRating(long WorkId, int Value, bool Dnf, string Title, WorkRef Ref);

public static class UserRatings
{
    /// <summary>Оценки пользователя с внешним id книги — вход сервиса выдачи.</summary>
    public static async Task<List<UserRating>> LoadAsync(NpgsqlConnection c, long userId) =>
        (await c.QueryAsync<(long WorkId, short Value, bool Dnf, string Title, string Source, string Ext)>("""
            SELECT r.work_id, r.value, coalesce(s.status = 'dnf', false), w.title, src.code, x.external_id
            FROM ratings r JOIN works w ON w.id = r.work_id
            JOIN external_ids x ON x.internal_id = r.work_id AND x.entity_type = 'work'
            JOIN sources src ON src.id = x.source_id
            LEFT JOIN shelves s ON s.user_id = r.user_id AND s.work_id = r.work_id
            WHERE r.user_id = @userId ORDER BY r.work_id
            """, new { userId }))
        .Select(r => new UserRating(r.WorkId, r.Value, r.Dnf, r.Title, new WorkRef(r.Source, r.Ext))).ToList();

    public static List<ServiceRating> ForService(IEnumerable<UserRating> ratings) =>
        ratings.Select(r => new ServiceRating(r.Ref.Source, r.Ref.Id, r.Value, r.Dnf, r.Title)).ToList();

    /// <summary>Внешние id → внутренние (книги, которых нет в каталоге БД, пропускаются).</summary>
    public static async Task<Dictionary<WorkRef, long>> IdsAsync(NpgsqlConnection c, IEnumerable<WorkRef> refs)
    {
        var list = refs.Distinct().ToList();
        if (list.Count == 0) return [];
        return (await c.QueryAsync<(string Source, string Ext, long Id)>("""
            SELECT q.source, q.ext, x.internal_id FROM unnest(@sources, @exts) AS q(source, ext)
            JOIN sources s ON s.code = q.source
            JOIN external_ids x ON x.source_id = s.id AND x.entity_type = 'work' AND x.external_id = q.ext
            """, new { sources = list.Select(r => r.Source).ToArray(), exts = list.Select(r => r.Id).ToArray() }))
            .ToDictionary(r => new WorkRef(r.Source, r.Ext), r => r.Id);
    }

    public static async Task<WorkRef?> RefAsync(NpgsqlConnection c, long workId) =>
        (await c.QueryAsync<(string Source, string Ext)>("""
            SELECT s.code, x.external_id FROM external_ids x JOIN sources s ON s.id = x.source_id
            WHERE x.entity_type = 'work' AND x.internal_id = @workId LIMIT 1
            """, new { workId })).Select(r => new WorkRef(r.Source, r.Ext)).Cast<WorkRef?>().FirstOrDefault();
}
