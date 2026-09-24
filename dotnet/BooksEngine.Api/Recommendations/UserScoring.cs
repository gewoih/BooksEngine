using Dapper;
using Npgsql;

namespace BooksEngine.Api.Recommendations;

public static class UserScoring
{
    /// <summary>Оценки пользователя (с флагом «бросил») + исключения для оценённых книг → расчёт смеси.</summary>
    public static async Task<Scoring> ForUserAsync(NpgsqlConnection c, ModelSnapshot m, long userId)
    {
        var ratings = (await c.QueryAsync<(long WorkId, short Value, bool Dnf)>("""
            SELECT r.work_id, r.value, coalesce(s.status = 'dnf', false) FROM ratings r
            LEFT JOIN shelves s ON s.user_id = r.user_id AND s.work_id = r.work_id
            WHERE r.user_id = @userId
            """, new { userId })).Select(r => new RatingInput(r.WorkId, r.Value, r.Dnf)).ToList();
        var excl = (await c.QueryAsync<(long Rated, long Excluded, string Reason)>("""
            SELECT e.rated_work_id, e.excluded_work_id, e.reason FROM work_exclusions e
            JOIN ratings r ON r.work_id = e.rated_work_id AND r.user_id = @userId
            """, new { userId })).Select(e => new ExclusionInput(e.Rated, e.Excluded, e.Reason == "series")).ToList();
        return Recommender.Score(m, ratings, excl);
    }
}
