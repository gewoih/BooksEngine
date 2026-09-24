using MathNet.Numerics.LinearAlgebra;

namespace BooksEngine.Api.Recommendations;

/// <summary>Dnf — недочитана: во входе EASE вес DnfInput (0), в ALS — как оценка (1★), mix.py.</summary>
public sealed record RatingInput(long WorkId, double Rating, bool Dnf = false);
public sealed record ExclusionInput(long RatedWorkId, long ExcludedWorkId, bool Series);
public sealed record Pick(long WorkId, double Score, int ChancePct, IReadOnlyList<long> Because, long? Despite);

/// <summary>Промежуточный расчёт одного человека: баллы смеси по позициям EASE и всё для шанса и объяснения.</summary>
public sealed class Scoring
{
    internal int[] InCols = [];
    internal double[] R = [];
    internal bool[] Dnf = [];
    internal double[][] G = [];                // вклад книги входа в вектор человека: w_i·A⁻¹y_i (explain.py)
    internal double[] GMean = [];              // G_i · среднее Y по книгам EASE
    internal double[] SAls = [], SEase = [];   // до нормировки, по позициям EASE
    internal double[] Mix = [];                // −∞ — не кандидат (вход, начатая серия)
    internal HashSet<int> RatedAlready = [];   // позиции: убрать из выдачи, но не из места для шанса
    internal double[] ValidSorted = [];
    internal int KLike;
    public int UsedRatings => InCols.Length;
}

/// <summary>
/// Выдача смеси — повтор Python: `Mix.score` (mix.py), `recommend` (recommend.py), `Chance.predict` (chance.py),
/// `explain.contributions` / `reason` (explain.py). Сверяется с Python эталонным тестом (GoldenTests).
/// Где Python округляет до float32, здесь то же — иначе близкие баллы меняются местами.
/// </summary>
public static class Recommender
{
    public static Scoring Score(ModelSnapshot m, IReadOnlyList<RatingInput> ratings, IReadOnlyCollection<ExclusionInput> exclusions)
    {
        var p = m.Params;
        var inputs = ratings.Where(r => m.ColByWork.ContainsKey(r.WorkId))
            .Select(r => (Col: m.ColByWork[r.WorkId], r.Rating, r.Dnf)).OrderBy(t => t.Col).ToArray();
        var s = new Scoring
        {
            InCols = inputs.Select(t => t.Col).ToArray(), R = inputs.Select(t => t.Rating).ToArray(),
            Dnf = inputs.Select(t => t.Dnf).ToArray(),
        };
        if (inputs.Length == 0) return s;
        int k = m.Factors, nE = m.NEase, n = inputs.Length;

        // ALS fold-in (als.py, fold_in_system): A = YᵀY + Yuᵀ·diag(m)·Yu + λI, A·x = Yuᵀ·((1 + m)·p)
        var A = Matrix<double>.Build.DenseOfArray((double[,])m.YtY.Clone());
        var b = Vector<double>.Build.Dense(k);
        var wv = new double[n];
        for (int i = 0; i < n; i++)
        {
            var r = inputs[i].Rating;
            var neg = p.AlsNegRule == "le2" && r <= 2.0;
            var mw = neg ? p.AlsNegWeight : p.AlsAlpha * r;
            wv[i] = (1.0 + mw) * (neg ? -1.0 : 1.0);
            var y = m.Row(inputs[i].Col);
            for (int a = 0; a < k; a++)
            {
                b[a] += wv[i] * y[a];
                if (mw != 0) for (int c = 0; c < k; c++) A[a, c] += mw * y[a] * y[c];
            }
        }
        for (int a = 0; a < k; a++) A[a, a] += p.AlsRegularization;
        var chol = A.Cholesky();
        var x = chol.Solve(b);

        s.SAls = new double[nE];
        for (int t = 0; t < nE; t++)
        {
            var y = m.Row(m.EaseCols[t]);
            double dot = 0;
            for (int f = 0; f < k; f++) dot += x[f] * y[f];
            s.SAls[t] = (float)dot;                  // als.score → float32
        }

        // Для объяснения: G_i = w_i·A⁻¹y_i — вклад книги i в вектор человека
        s.G = new double[n][];
        s.GMean = new double[n];
        for (int i = 0; i < n; i++)
        {
            var y = m.Row(inputs[i].Col);
            var yi = Vector<double>.Build.Dense(k);
            for (int f = 0; f < k; f++) yi[f] = y[f];
            s.G[i] = (chol.Solve(yi) * wv[i]).ToArray();
            for (int f = 0; f < k; f++) s.GMean[i] += s.G[i][f] * m.YEaseMean[f];
        }

        // EASE: Σ v_i·B[i, ·], v — вес оценки (1★ −2 … 5★ +2, недочитана — DnfInput); scipy копит во float32
        var ease = new float[nE];
        for (int i = 0; i < n; i++)
        {
            var pos = m.EasePosByCol[inputs[i].Col];
            var v = (float)EaseInput(p, inputs[i].Rating, inputs[i].Dnf);
            if (pos < 0 || v == 0) continue;
            for (int e = m.EaseRowPtr[pos]; e < m.EaseRowPtr[pos + 1]; e++) ease[m.EaseTo[e]] += v * m.EaseWeight[e];
        }
        s.SEase = ease.Select(v => (double)v).ToArray();

        var zA = Z(s.SAls);
        var zE = Z(s.SEase);
        s.Mix = new double[nE];
        for (int t = 0; t < nE; t++) s.Mix[t] = (float)(p.AlsWeight * zA[t] + (1 - p.AlsWeight) * zE[t]);   // mix.score → float32

        foreach (var col in s.InCols)
            if (m.EasePosByCol[col] is var pos and >= 0) s.Mix[pos] = double.NegativeInfinity;
        foreach (var ex in exclusions)
        {
            if (!m.ColByWork.TryGetValue(ex.ExcludedWorkId, out var col) || m.EasePosByCol[col] < 0) continue;
            if (ex.Series) s.Mix[m.EasePosByCol[col]] = double.NegativeInfinity;
            else s.RatedAlready.Add(m.EasePosByCol[col]);
        }
        s.ValidSorted = s.Mix.Where(double.IsFinite).Order().ToArray();
        s.KLike = inputs.Count(t => Rounded(t.Rating) >= 4);
        return s;
    }

    public static IReadOnlyList<Pick> Top(ModelSnapshot m, Scoring s, int n)
    {
        if (s.UsedRatings == 0) return [];
        var picked = Enumerable.Range(0, m.NEase)
            .Where(t => double.IsFinite(s.Mix[t]))
            .OrderByDescending(t => s.Mix[t]).ThenBy(t => t)       // позиции EASE растут со столбцом
            .Where(t => !s.RatedAlready.Contains(t))
            .Take(n).ToArray();
        return picked.Select(t =>
        {
            var (because, despite) = Explain(m, s, t);
            return new Pick(m.WorkIdByCol[m.EaseCols[t]], s.Mix[t], ChancePct(m, s, t), because, despite);
        }).ToList();
    }

    public static int? ChanceFor(ModelSnapshot m, Scoring s, long workId)
    {
        if (s.UsedRatings == 0 || !m.ColByWork.TryGetValue(workId, out var col)) return null;
        var t = m.EasePosByCol[col];
        return t >= 0 && double.IsFinite(s.Mix[t]) ? ChancePct(m, s, t) : null;
    }

    // chance.py: место в личном рейтинге (выше + 1) / кандидатов и щедрость человека, логистическая регрессия
    private static int ChancePct(ModelSnapshot m, Scoring s, int t)
    {
        var p = m.Params;
        var nValid = s.ValidSorted.Length;
        var above = nValid - UpperBound(s.ValidSorted, s.Mix[t]);
        var pct = (above + 1.0) / Math.Max(nValid, 1);
        var own = Math.Clamp((s.KLike + p.ChancePrior * p.ChanceP0) / (s.UsedRatings + p.ChancePrior), 1e-6, 1 - 1e-6);
        var z = p.ChanceCoef[0] + p.ChanceCoef[1] * Math.Log10(pct) + p.ChanceCoef[2] * Math.Log(own / (1 - own));
        return (int)Math.Round(100.0 / (1 + Math.Exp(-z)));     // как int(round(·)) Python: половины — к чётному
    }

    // explain.py: вклад книги входа i в балл t; сумма по i равна баллу смеси
    private static (IReadOnlyList<long> Because, long? Despite) Explain(ModelSnapshot m, Scoring s, int t)
    {
        var p = m.Params;
        int k = m.Factors, n = s.InCols.Length;
        var sdA = Math.Max(Std(s.SAls), 1e-9);
        var sdE = Math.Max(Std(s.SEase), 1e-9);
        var yt = m.Row(m.EaseCols[t]);
        var c = new double[n];
        for (int i = 0; i < n; i++)
        {
            double cA = -s.GMean[i];
            for (int f = 0; f < k; f++) cA += s.G[i][f] * yt[f];

            double cE = 0;
            var pos = m.EasePosByCol[s.InCols[i]];
            var v = EaseInput(p, s.R[i], s.Dnf[i]);
            if (pos >= 0 && v != 0)
            {
                double rowSum = 0, bt = 0;
                for (int e = m.EaseRowPtr[pos]; e < m.EaseRowPtr[pos + 1]; e++)
                {
                    rowSum += m.EaseWeight[e];
                    if (m.EaseTo[e] == t) bt = m.EaseWeight[e];
                }
                cE = v * bt - v * rowSum / m.NEase;
            }
            c[i] = p.AlsWeight * cA / sdA + (1 - p.AlsWeight) * cE / sdE;
        }
        var order = Enumerable.Range(0, n).OrderByDescending(i => c[i]).ToArray();    // устойчивая, как kind="stable"
        var leader = n > 0 ? c[order[0]] : 0.0;
        if (leader <= 0) return ([], null);
        var because = order.Take(p.MaxBecause).Where(i => c[i] >= p.MinOfLeader * leader)
            .Select(i => m.WorkIdByCol[s.InCols[i]]).ToList();
        var low = Enumerable.Range(0, n).MinBy(i => c[i]);                             // первый минимум, как argmin
        long? despite = -c[low] >= p.DespiteOfLeader * leader ? m.WorkIdByCol[s.InCols[low]] : null;
        return (because, despite);
    }

    private static double EaseInput(ModelParams p, double r, bool dnf) => dnf ? p.DnfInput : p.EaseInput[Rounded(r) - 1];
    private static int Rounded(double r) => (int)Math.Floor(r + 0.5);                  // metrics.rounded

    private static double[] Z(double[] v)
    {
        var mean = v.Average();
        var sd = Math.Max(Std(v), 1e-9);
        return v.Select(x => (x - mean) / sd).ToArray();
    }

    private static double Std(double[] v)
    {
        var mean = v.Average();
        return Math.Sqrt(v.Sum(x => (x - mean) * (x - mean)) / v.Length);             // ddof = 0, как numpy
    }

    private static int UpperBound(double[] sorted, double x)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi) { var mid = (lo + hi) / 2; if (sorted[mid] <= x) lo = mid + 1; else hi = mid; }
        return lo;
    }
}
