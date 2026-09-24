using BooksEngine.Api.Recommendations;
using Dapper;
using Npgsql;

namespace BooksEngine.Api.Tests;

/// <summary>
/// C# повторяет Python на выгруженной модели: рабочая БД booksengine, только чтение. Пропускается, если
/// Postgres не поднят или модель не выгружена (`uv run booksengine export-model`). Запускать после каждого экспорта.
/// </summary>
public sealed class GoldenTests
{
    private sealed class Golden
    {
        public string Profile { get; init; } = "";
        public int Rank { get; init; }
        public long WorkId { get; init; }
        public double Score { get; init; }
        public int Chance { get; init; }
        public long[] Because { get; init; } = [];
        public long? Despite { get; init; }
    }

    [Fact]
    public async Task Recommendations_match_python_on_golden_profiles()
    {
        await using var ds = NpgsqlDataSource.Create(
            new NpgsqlConnectionStringBuilder(Db.ConnectionString.FromEnvironment()) { Timeout = 3 }.ConnectionString);
        ModelSnapshot? m = null;
        var available = true;
        try { m = await ModelLoader.LoadAsync(ds, TestContext.Current.CancellationToken); }
        catch (NpgsqlException) { available = false; }
        Assert.SkipUnless(available, "PostgreSQL недоступен");
        Assert.SkipWhen(m is null, "модель не выгружена: uv run booksengine export-model");

        DefaultTypeMap.MatchNamesWithUnderscores = true;
        await using var c = await ds.OpenConnectionAsync(TestContext.Current.CancellationToken);
        var inputs = (await c.QueryAsync<(string Profile, long WorkId, double Rating, bool Dnf)>(
            "SELECT profile, work_id, rating, dnf FROM golden_inputs")).ToLookup(r => r.Profile);
        var golden = (await c.QueryAsync<Golden>(
            "SELECT profile, rank, work_id, score, chance, because, despite FROM golden_recommendations ORDER BY profile, rank"))
            .GroupBy(g => g.Profile).ToList();
        Assert.NotEmpty(golden);

        foreach (var g in golden)
        {
            var ratings = inputs[g.Key].Select(r => new RatingInput(r.WorkId, r.Rating, r.Dnf)).ToList();
            var excl = (await c.QueryAsync<(long Rated, long Excluded, string Reason)>(
                    "SELECT rated_work_id, excluded_work_id, reason FROM work_exclusions WHERE rated_work_id = ANY(@ids)",
                    new { ids = ratings.Select(r => r.WorkId).ToArray() }))
                .Select(e => new ExclusionInput(e.Rated, e.Excluded, e.Reason == "series")).ToList();
            var want = g.ToList();
            var got = Recommender.Top(m!, Recommender.Score(m!, ratings, excl), want.Count);

            Assert.Equal(want.Count, got.Count);
            for (int i = 0; i < want.Count; i++)
            {
                var (w, r) = (want[i], got[i]);
                // соседи с почти равным баллом (float32 в Python) могут поменяться местами — это не расхождение
                var tie = i + 1 < want.Count && Math.Abs(want[i].Score - want[i + 1].Score) < 1e-5
                          || i > 0 && Math.Abs(want[i].Score - want[i - 1].Score) < 1e-5;
                if (!tie) Assert.True(w.WorkId == r.WorkId, $"{g.Key} #{w.Rank}: Python {w.WorkId}, C# {r.WorkId}");
                var same = got.Single(x => x.WorkId == w.WorkId);
                Assert.True(Math.Abs(w.Score - same.Score) < 1e-4, $"{g.Key} #{w.Rank}: балл {w.Score} против {same.Score}");
                Assert.True(Math.Abs(w.Chance - same.ChancePct) <= 1, $"{g.Key} #{w.Rank}: шанс {w.Chance} против {same.ChancePct}");
                Assert.True(w.Because.SequenceEqual(same.Because),
                    $"{g.Key} #{w.Rank}: «потому что» {string.Join(",", w.Because)} против {string.Join(",", same.Because)}");
                Assert.True(w.Despite == same.Despite, $"{g.Key} #{w.Rank}: «несмотря на» {w.Despite} против {same.Despite}");
            }
        }
    }
}
