using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Recommendations;

/// <summary>Сравнение с выдачей при прежних оценках: места книг до (размер списка → work_id → место) и что
/// поменялось в оценках.</summary>
public sealed record RankComparison(IReadOnlyDictionary<int, Dictionary<long, int>> Before, IReadOnlyList<RatingChange> Changes);

/// <summary>Что поменялось в оценках: Before/After — «5», «dnf» или null (не было оценки / удалена).</summary>
public sealed record RatingChange(long WorkId, string? Before, string? After);

/// <summary>
/// Слепки выдачи по наборам оценок (recommendation_snapshots): у каждого набора оценок — свои места книг. Хранятся
/// текущий набор и прежний; выдача сравнивается с прежним, пока оценки не поменяются снова. Места — по каждому размеру
/// списка (10, 20, 50): правила списка зависят от размера, и сравнивается список того же размера.
/// </summary>
public static class RankChanges
{
    public static string Label(UserRating r) => r.Dnf ? "dnf" : r.Value.ToString();

    public static string Hash(IReadOnlyDictionary<long, string> ratings) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
            string.Join(";", ratings.OrderBy(p => p.Key).Select(p => $"{p.Key}:{p.Value}")))));

    /// <summary>Записать места текущей выдачи и вернуть сравнение с прежним набором оценок (нет прежнего — null).</summary>
    public static async Task<RankComparison?> SaveAndCompareAsync(NpgsqlConnection c, long userId,
        IReadOnlyDictionary<long, string> ratings, IReadOnlyDictionary<int, Dictionary<long, int>> ranks)
    {
        var hash = Hash(ratings);
        var ranksJson = JsonSerializer.Serialize(ranks);
        await using var tx = await c.BeginTransactionAsync();
        // один пользователь — один пересчёт слепков за раз (две вкладки не создадут два «текущих»)
        await c.ExecuteAsync("SELECT pg_advisory_xact_lock(hashtext('booksengine.snapshots'), @u::int)",
            new { u = (int)(userId % int.MaxValue) }, tx);
        var last = (await c.QueryAsync<(long Id, string Hash, string Ratings, string Ranks)>("""
            SELECT id, ratings_hash, ratings::text, ranks::text FROM recommendation_snapshots
            WHERE user_id = @userId ORDER BY id DESC LIMIT 2
            """, new { userId }, tx)).ToList();
        (long Id, string Hash, string Ratings, string Ranks)? prev;
        if (last.Count > 0 && last[0].Hash == hash)
        {
            // тот же набор оценок: места обновляются (модель могла смениться), сравнение — с прежним набором
            await c.ExecuteAsync("UPDATE recommendation_snapshots SET ranks = @ranksJson::jsonb WHERE id = @id",
                new { ranksJson, id = last[0].Id }, tx);
            prev = last.Count > 1 ? last[1] : null;
        }
        else
        {
            var id = await c.ExecuteScalarAsync<long>("""
                INSERT INTO recommendation_snapshots (user_id, ratings_hash, ratings, ranks)
                VALUES (@userId, @hash, @ratingsJson::jsonb, @ranksJson::jsonb) RETURNING id
                """, new { userId, hash, ratingsJson = JsonSerializer.Serialize(ratings), ranksJson }, tx);
            prev = last.Count > 0 ? last[0] : null;
            await c.ExecuteAsync("DELETE FROM recommendation_snapshots WHERE user_id = @userId AND id <> ALL(@keep)",
                new { userId, keep = prev is { } p ? new[] { id, p.Id } : [id] }, tx);
        }
        await tx.CommitAsync();
        if (prev is not { } before) return null;

        var oldRatings = JsonSerializer.Deserialize<Dictionary<long, string>>(before.Ratings)!;
        var changes = ratings.Keys.Union(oldRatings.Keys).Order()
            .Select(w => new RatingChange(w, oldRatings.GetValueOrDefault(w), ratings.GetValueOrDefault(w)))
            .Where(ch => ch.Before != ch.After).ToList();
        return new RankComparison(JsonSerializer.Deserialize<Dictionary<int, Dictionary<long, int>>>(before.Ranks)!, changes);
    }
}
