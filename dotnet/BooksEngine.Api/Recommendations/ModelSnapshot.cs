namespace BooksEngine.Api.Recommendations;

/// <summary>
/// Модель в памяти: столбцы CF-ядра, векторы ALS (Y), соседи EASE (B по строкам «откуда»). Неизменяема —
/// новая модель заменяет старую целиком (ModelStore). Столбцы — порядок Python (Goodreads-id по возрастанию):
/// по ним разрешаются равные баллы, как `np.argsort(kind="stable")`.
/// </summary>
public sealed class ModelSnapshot
{
    public required string Fingerprint { get; init; }
    public required ModelParams Params { get; init; }
    public required long[] WorkIdByCol { get; init; }
    public required Dictionary<long, int> ColByWork { get; init; }
    public required int Factors { get; init; }
    public required float[] Y { get; init; }              // [col · Factors + f]
    public required double[,] YtY { get; init; }
    public required double[] YEaseMean { get; init; }     // среднее Y по книгам EASE — сдвиг вклада (explain.py)
    public required int[] EaseCols { get; init; }         // позиция EASE → столбец, по возрастанию
    public required int[] EasePosByCol { get; init; }     // столбец → позиция EASE или −1
    public required int[] EaseRowPtr { get; init; }       // соседи позиции p: [RowPtr[p], RowPtr[p + 1])
    public required int[] EaseTo { get; init; }           // по возрастанию внутри строки
    public required float[] EaseWeight { get; init; }

    public int NEase => EaseCols.Length;
    public bool InEase(long workId) => ColByWork.TryGetValue(workId, out var c) && EasePosByCol[c] >= 0;
    public ReadOnlySpan<float> Row(int col) => Y.AsSpan(col * Factors, Factors);

    public static ModelSnapshot Create(string fingerprint, ModelParams p, long[] workIdByCol, int factors,
        float[] y, int[] easeCols, int[] easeRowPtr, int[] easeTo, float[] easeWeight)
    {
        int n = workIdByCol.Length, k = factors;
        var yty = new double[k, k];
        for (int c = 0; c < n; c++)
            for (int a = 0; a < k; a++)
            {
                double ya = y[c * k + a];
                for (int b = a; b < k; b++) yty[a, b] += ya * y[c * k + b];
            }
        for (int a = 0; a < k; a++) for (int b = 0; b < a; b++) yty[a, b] = yty[b, a];

        var mean = new double[k];
        foreach (var c in easeCols) for (int f = 0; f < k; f++) mean[f] += y[c * k + f];
        for (int f = 0; f < k; f++) mean[f] /= easeCols.Length;

        var posByCol = Enumerable.Repeat(-1, n).ToArray();
        for (int t = 0; t < easeCols.Length; t++) posByCol[easeCols[t]] = t;

        return new ModelSnapshot
        {
            Fingerprint = fingerprint, Params = p, WorkIdByCol = workIdByCol,
            ColByWork = workIdByCol.Select((w, c) => (w, c)).ToDictionary(t => t.w, t => t.c),
            Factors = k, Y = y, YtY = yty, YEaseMean = mean, EaseCols = easeCols, EasePosByCol = posByCol,
            EaseRowPtr = easeRowPtr, EaseTo = easeTo, EaseWeight = easeWeight,
        };
    }
}
