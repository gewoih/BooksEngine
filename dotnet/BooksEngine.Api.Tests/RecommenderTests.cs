using BooksEngine.Api.Recommendations;

namespace BooksEngine.Api.Tests;

public sealed class RecommenderTests
{
    // 5 книг (id 10..14), 2 фактора, все в EASE. als_weight = 0: выдача — только EASE, её легко посчитать руками.
    private static ModelSnapshot Model(double alsWeight = 0.0) => ModelSnapshot.Create("t",
        new ModelParams(alsWeight, [-2, -1, 0, 1, 2], 0.0, 1.0, 0.1, "le2", 0.0, [-0.9, -0.46, 0.85], 3.0, 0.72, 3, 0.25, 0.5),
        workIdByCol: [10, 11, 12, 13, 14], factors: 2,
        y: [1, 0, 0.9f, 0.1f, 0.2f, 1, 0, 1, 0.7f, 0.7f],
        easeCols: [0, 1, 2, 3, 4],
        // CSR по «откуда»: 10 → 11 (0.9), 10 → 12 (0.5), 10 → 14 (0.6); 13 → 12 (−0.8)
        easeRowPtr: [0, 3, 3, 3, 4, 4], easeTo: [1, 2, 4, 2], easeWeight: [0.9f, 0.5f, 0.6f, -0.8f]);

    [Fact]
    public void Ease_only_orders_by_weighted_neighbours_and_skips_rated()
    {
        var m = Model();
        var s = Recommender.Score(m, [new(10, 5), new(13, 1)], []);
        // EASE: 5★ → +2, 1★ → −2. Балл 11 = 1.8, 12 = 2·0.5 + (−2)(−0.8) = 2.6, 14 = 1.2; 10 и 13 — вход
        Assert.Equal([12L, 11, 14], Recommender.Top(m, s, 10).Select(p => p.WorkId));
        Assert.Equal(2, s.UsedRatings);
    }

    [Fact]
    public void Series_exclusion_removes_candidate_and_rated_filter_only_hides_it()
    {
        var m = Model();
        var series = Recommender.Score(m, [new(10, 5)], [new(10, 11, Series: true)]);
        var rated = Recommender.Score(m, [new(10, 5)], [new(10, 11, Series: false)]);
        Assert.DoesNotContain(11L, Recommender.Top(m, series, 10).Select(p => p.WorkId));
        Assert.DoesNotContain(11L, Recommender.Top(m, rated, 10).Select(p => p.WorkId));
        // «уже оценено» не выбрасывает книгу из места в рейтинге: у 12 выше неё остаётся 11 → шанс ниже, чем с серией
        Assert.True(Recommender.ChanceFor(m, rated, 12) < Recommender.ChanceFor(m, series, 12));
    }

    [Fact]
    public void Explanation_names_the_book_that_pulled_up_and_despite_the_one_that_pulled_down()
    {
        var m = Model();
        var top = Recommender.Top(m, Recommender.Score(m, [new(10, 5), new(13, 1)], []), 10);
        // вклад со сдвигом на среднее строки EASE по 5 книгам (explain.py): 10 → 12: 2·0.5 − 2·2.0/5 = 0.2;
        // 13 → 12: (−2)(−0.8) − (−2)(−0.8)/5 = 1.28 — 1★ у 13 при отрицательном весе тянет вверх. 0.2 < 25% лидера
        Assert.Equal([13L], top.Single(p => p.WorkId == 12).Because);
        // 10 → 11: 2·0.9 − 0.8 = 1.0; 13 → 11: 0 − 0.32 = −0.32 — тянет вниз, но меньше половины лидера
        var p11 = top.Single(p => p.WorkId == 11);
        Assert.Equal([10L], p11.Because);
        Assert.Null(p11.Despite);
    }

    [Fact]
    public void Dnf_is_zero_for_ease_but_negative_for_als()
    {
        var m = Model();                                       // als_weight = 0 — выдача только EASE
        var oneStar = Recommender.Score(m, [new(10, 5), new(13, 1)], []);
        var dnf = Recommender.Score(m, [new(10, 5), new(13, 1, Dnf: true)], []);
        // 1★ у 13 тянет 12 вверх (вес 13 → 12 отрицательный); недочитанная 13 весит 0 — 12 держится только на 10
        var top1 = Recommender.Top(m, oneStar, 10).Single(p => p.WorkId == 12);
        var top2 = Recommender.Top(m, dnf, 10).Single(p => p.WorkId == 12);
        Assert.Contains(13L, top1.Because);
        Assert.DoesNotContain(13L, top2.Because);
        Assert.Equal([11L, 14, 12], Recommender.Top(m, dnf, 10).Select(p => p.WorkId));   // 1.8, 1.2, 1.0
    }

    [Fact]
    public void Only_three_star_or_unknown_books_give_finite_chances_not_nan()
    {
        var m = Model(alsWeight: 0.5);
        var s = Recommender.Score(m, [new(12, 3), new(999, 5)], []);   // 3★ — вес EASE 0; 999 — не в модели
        var top = Recommender.Top(m, s, 10);
        Assert.Equal(1, s.UsedRatings);
        Assert.NotEmpty(top);
        Assert.All(top, p => Assert.InRange(p.ChancePct, 0, 100));
        Assert.All(top, p => Assert.False(double.IsNaN(p.Score)));
    }

    [Fact]
    public void No_ratings_in_model_gives_empty_list()
    {
        var m = Model();
        Assert.Empty(Recommender.Top(m, Recommender.Score(m, [new(999, 5)], []), 10));
    }
}
