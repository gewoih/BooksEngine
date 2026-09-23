namespace BooksEngine.Db.Entities;

/// <summary>Источник данных (goodreads, позже другие). Внешние id каталога привязаны к источнику.</summary>
public class Source
{
    public int Id { get; set; }
    public required string Code { get; set; }
}

/// <summary>Внешний id сущности каталога в источнике → внутренний id. Goodreads-id не становится первичным ключом.</summary>
public class ExternalId
{
    public int SourceId { get; set; }
    public required string EntityType { get; set; }
    public required string Value { get; set; }
    public long InternalId { get; set; }
}

/// <summary>Журнал загрузок каталога из источника: по контрольной сумме манифеста повторная загрузка распознаётся.</summary>
public class SourceLoad
{
    public long Id { get; set; }
    public int SourceId { get; set; }
    public required string ManifestSha256 { get; set; }
    public required string Counts { get; set; }
    public DateTime LoadedAt { get; set; }
}

/// <summary>Произведение — все издания одного текста. Оценки и рекомендации — на этом уровне.</summary>
public class Work
{
    public long Id { get; set; }
    public required string Title { get; set; }
    public string? OriginalTitle { get; set; }
    public long? BestEditionId { get; set; }
    public int? PublicationYear { get; set; }
    public string? LanguageCode { get; set; }
    public string? Description { get; set; }
    public bool IsCollection { get; set; }
    /// <summary>Входит в CF-ядро (k-core) — у модели есть для него вектор.</summary>
    public bool InCf { get; set; }
    public int CfRatings { get; set; }
    public double? CfMeanRating { get; set; }
    public long? GrRatingsCount { get; set; }
    public long? GrRatingsSum { get; set; }

    public Edition? BestEdition { get; set; }
    public List<Edition> Editions { get; set; } = [];
}

/// <summary>Издание — конкретная публикация произведения: перевод, переплёт, аудиокнига.</summary>
public class Edition
{
    public long Id { get; set; }
    public long WorkId { get; set; }
    /// <summary>В источнике у 3 изданий названия нет; название произведения есть всегда.</summary>
    public string? Title { get; set; }
    public string? TitleWithoutSeries { get; set; }
    public string? Isbn { get; set; }
    public string? Isbn13 { get; set; }
    public string? Asin { get; set; }
    public string? KindleAsin { get; set; }
    public string? LanguageCode { get; set; }
    public string? CountryCode { get; set; }
    public string? Format { get; set; }
    public bool? IsEbook { get; set; }
    public int? NumPages { get; set; }
    public int? PublicationYear { get; set; }
    public string? Publisher { get; set; }
    public long? RatingsCount { get; set; }
    public double? AverageRating { get; set; }
    public string? ImageUrl { get; set; }
    public string? Url { get; set; }

    public Work Work { get; set; } = null!;
}

public class Author
{
    public long Id { get; set; }
    /// <summary>В источнике у 2 авторов имени нет, но на них ссылаются произведения.</summary>
    public string? Name { get; set; }
    public double? AverageRating { get; set; }
    public long? RatingsCount { get; set; }
}

public class WorkAuthor
{
    public long WorkId { get; set; }
    public long AuthorId { get; set; }
    /// <summary>Роль из источника (Illustrator, Translator…); пусто — основной автор.</summary>
    public string? Role { get; set; }
    public short Position { get; set; }
}

public class Genre
{
    public int Id { get; set; }
    public required string Name { get; set; }
}

public class WorkGenre
{
    public long WorkId { get; set; }
    public int GenreId { get; set; }
    public int Votes { get; set; }
    public double Share { get; set; }
}
