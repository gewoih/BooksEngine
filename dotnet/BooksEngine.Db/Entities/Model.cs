namespace BooksEngine.Db.Entities;

// Данные приложения сверх каталога. Наполняет Python (`booksengine export-app`) одной транзакцией, перезаписывая
// всё. Выдачу считает сервис `booksengine serve`, её модели в БД нет.

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
