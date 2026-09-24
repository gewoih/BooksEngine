using System.Text.Json;
using Npgsql;
using NpgsqlTypes;

namespace BooksEngine.Api.Recommendations;

/// <summary>Чтение выгруженной модели из БД (`booksengine export-model`). Нет model_meta — модели нет (null).</summary>
public static class ModelLoader
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    public static async Task<ModelSnapshot?> LoadAsync(NpgsqlDataSource ds, CancellationToken ct)
    {
        await using var c = await ds.OpenConnectionAsync(ct);
        await using var tx = await c.BeginTransactionAsync(System.Data.IsolationLevel.RepeatableRead, ct);  // один снимок
        string fp, json;
        await using (var cmd = new NpgsqlCommand("SELECT fingerprint, params::text FROM model_meta WHERE id = 1", c, tx))
        await using (var r = await cmd.ExecuteReaderAsync(ct))
        {
            if (!await r.ReadAsync(ct)) return null;
            (fp, json) = (r.GetString(0), r.GetString(1));
        }
        var p = JsonSerializer.Deserialize<ModelParams>(json, Json)!;

        var workIds = new List<long>();
        var easeByCol = new List<(int Pos, int Col)>();
        var y = new List<float>();
        int factors = 0;
        // вектор читается как real[] — штатное приведение pgvector, без типа vector в Npgsql
        await using (var cmd = new NpgsqlCommand(
            "SELECT col, work_id, ease_pos, embedding::real[] FROM work_embeddings ORDER BY col", c, tx))
        await using (var r = await cmd.ExecuteReaderAsync(ct))
            while (await r.ReadAsync(ct))
            {
                var col = r.GetInt32(0);
                if (col != workIds.Count) throw new InvalidOperationException($"work_embeddings: пропущен столбец {workIds.Count}");
                workIds.Add(r.GetInt64(1));
                if (!r.IsDBNull(2)) easeByCol.Add((r.GetInt32(2), col));
                var v = r.GetFieldValue<float[]>(3);
                factors = v.Length;
                y.AddRange(v);
            }
        var easeCols = easeByCol.OrderBy(t => t.Pos).Select(t => t.Col).ToArray();

        var rowPtr = new int[easeCols.Length + 1];
        var to = new List<int>();
        var w = new List<float>();
        await using (var exp = await c.BeginBinaryExportAsync(
            "COPY (SELECT from_pos, to_pos, weight FROM ease_weights ORDER BY from_pos, to_pos) TO STDOUT (FORMAT BINARY)", ct))
            while (await exp.StartRowAsync(ct) != -1)
            {
                var from = await exp.ReadAsync<int>(NpgsqlDbType.Integer, ct);
                to.Add(await exp.ReadAsync<int>(NpgsqlDbType.Integer, ct));
                w.Add(await exp.ReadAsync<float>(NpgsqlDbType.Real, ct));
                rowPtr[from + 1]++;
            }
        for (int i = 0; i < easeCols.Length; i++) rowPtr[i + 1] += rowPtr[i];
        await tx.CommitAsync(ct);
        return ModelSnapshot.Create(fp, p, workIds.ToArray(), factors, y.ToArray(), easeCols, rowPtr, to.ToArray(), w.ToArray());
    }
}
