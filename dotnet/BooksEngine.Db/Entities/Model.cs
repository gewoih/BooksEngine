using Pgvector;

namespace BooksEngine.Db.Entities;

// Таблицы онлайн-модели. Наполняет Python (`booksengine export-model`) одной транзакцией, перезаписывая
// всё: версий нет. Читает API (`ModelLoader`).

/// <summary>Параметры выгруженной модели — одна строка (Id = 1): смесь, шанс, пороги объяснения (JSON).</summary>
public class ModelMeta
{
    public int Id { get; set; }
    public required string Fingerprint { get; set; }
    public required string Params { get; set; }
    public DateTime ExportedAt { get; set; }
}

/// <summary>Книга CF-ядра в модели: столбец матрицы, позиция в EASE (null — вне 30 000) и вектор ALS.</summary>
public class WorkEmbedding
{
    public long WorkId { get; set; }
    public int Col { get; set; }
    public int? EasePos { get; set; }
    public required Vector Embedding { get; set; }
}

/// <summary>Вес EASE B[from, to] по позициям EASE: 500 самых сильных «откуда» на каждую книгу «куда».</summary>
public class EaseWeight
{
    public int FromPos { get; set; }
    public int ToPos { get; set; }
    public float Weight { get; set; }
}

public static class ExclusionReason
{
    /// <summary>Та же серия: книга не кандидат вовсе (и в месте для шанса не участвует).</summary>
    public const string Series = "series";
    /// <summary>Дубль, сборник или часть оценённого сборника: убирается только из выдачи.</summary>
    public const string Rated = "rated";
    public static readonly string[] All = [Series, Rated];
}

/// <summary>Оценил RatedWorkId → не советовать ExcludedWorkId.</summary>
public class WorkExclusion
{
    public long RatedWorkId { get; set; }
    public long ExcludedWorkId { get; set; }
    public required string Reason { get; set; }
}

/// <summary>Тень Goodreads (её work_id в источнике) → главное произведение. Для импорта CSV.</summary>
public class WorkMerge
{
    public required string ShadowExternalId { get; set; }
    public long MainWorkId { get; set; }
}

/// <summary>Обложка произведения: самое популярное издание с картинкой.</summary>
public class WorkCover
{
    public long WorkId { get; set; }
    public required string ImageUrl { get; set; }
}

/// <summary>Эталонный профиль: вход выдачи Python (оценка может быть дробной — среднее дублей; Dnf — недочитана).</summary>
public class GoldenInput
{
    public required string Profile { get; set; }
    public long WorkId { get; set; }
    public double Rating { get; set; }
    public bool Dnf { get; set; }
}

/// <summary>Эталонная выдача Python (`recommend`) — C# обязан её повторить.</summary>
public class GoldenRecommendation
{
    public required string Profile { get; set; }
    public int Rank { get; set; }
    public long WorkId { get; set; }
    public double Score { get; set; }
    public int Chance { get; set; }
    public long[] Because { get; set; } = [];
    public long? Despite { get; set; }
}
