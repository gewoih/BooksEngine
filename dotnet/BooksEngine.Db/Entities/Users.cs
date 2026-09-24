namespace BooksEngine.Db.Entities;

/// <summary>Пользователь приложения (не пользователь датасета).</summary>
public class AppUser
{
    public long Id { get; set; }
    public required string Name { get; set; }
    public DateTime CreatedAt { get; set; }
}

/// <summary>Оценка пользователя приложения, 1–5 (шкала Goodreads и модели).</summary>
public class Rating
{
    public long UserId { get; set; }
    public long WorkId { get; set; }
    public short Value { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}

public static class ShelfStatus
{
    public const string ToRead = "to_read";
    public const string Reading = "reading";
    public const string Read = "read";
    public const string Dnf = "dnf";
    public static readonly string[] All = [ToRead, Reading, Read, Dnf];
}

/// <summary>Статус книги у пользователя. Статусы взаимоисключающие, поэтому одна строка на пару (пользователь, произведение).</summary>
public class ShelfEntry
{
    public long UserId { get; set; }
    public long WorkId { get; set; }
    public required string Status { get; set; }
    public DateTime AddedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}
