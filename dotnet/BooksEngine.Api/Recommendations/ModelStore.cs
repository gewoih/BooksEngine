using Npgsql;

namespace BooksEngine.Api.Recommendations;

/// <summary>Текущая модель в памяти. Замена — целиком: запросы в полёте досчитывают на старой.</summary>
public sealed class ModelStore(NpgsqlDataSource ds, ILogger<ModelStore> log)
{
    private volatile ModelSnapshot? current;
    public ModelSnapshot? Current => current;

    public async Task<string?> ReloadAsync(CancellationToken ct)
    {
        var started = DateTime.UtcNow;
        current = await ModelLoader.LoadAsync(ds, ct);
        log.LogInformation("Модель {Fingerprint} загружена за {Seconds:F1} с", current?.Fingerprint ?? "(нет)",
            (DateTime.UtcNow - started).TotalSeconds);
        return current?.Fingerprint;
    }
}
