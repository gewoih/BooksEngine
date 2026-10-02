namespace BooksEngine.Api;

// Контракты API (JSON camelCase). Типы для фронта — из OpenAPI (`npm run gen:api` в web/).

public record UserDto(long Id, string Name);
public record CreateUserRequest(string Name);
public record SessionRequest(long UserId);

/// <summary>Title — по-русски, если перевод известен (works.ru_title), тогда TitleEn — английское; Author — тоже.</summary>
public record BookDto(long WorkId, string Title, string? TitleEn, string? Author, int? Year, string? CoverUrl,
                      int CfRatings, int? MyRating, bool MyDnf, bool InCore);
public record PageDto<T>(IReadOnlyList<T> Items, int Page, int PageSize, int Total);

/// <summary>Оценка 1–5 или «бросил» (Dnf = true, Value игнорируется): бросил = 1 + полка dnf.</summary>
public record RateRequest(int? Value, bool Dnf);
public record MyRatingDto(BookDto Book, DateTime UpdatedAt);

/// <summary>Книга выдачи: место в своём списке, шанс, подписи (почему советуется). PreviousRank — место при прежних
/// оценках (null — книги тогда в списке не было или сравнивать не с чем: IsNew это различает).</summary>
public record RecommendationDto(BookDto Book, int Rank, int ChancePct, IReadOnlyList<string> Why, int? PreviousRank,
                                bool IsNew);
public record RecommendationSectionDto(string Name, string? Note, IReadOnlyList<RecommendationDto> Items);
/// <summary>Changes — чем нынешние оценки отличаются от прежних, с которыми сравниваются места («Дюна: 5★»,
/// «Эмма: оценка удалена»); пусто — сравнивать не с чем.</summary>
public record RecommendationsDto(IReadOnlyList<RecommendationSectionDto> Sections, int UsedRatings, string ChanceLabel,
                                 string Legend, IReadOnlyList<string> Changes);
public record WorkDetailDto(BookDto Book, string? LargeCoverUrl, string? Description, IReadOnlyList<string> Authors,
                            IReadOnlyList<string> Genres, int? ChancePct, string? ChanceLabel, bool InModel,
                            IReadOnlyList<BookDto> Similar);

public record ImportIssue(int Line, string Book, string Reason);
public record ImportResultDto(int Added, int Updated, int Skipped, IReadOnlyList<ImportIssue> NotFound,
                              IReadOnlyList<ImportIssue> Errors, IReadOnlyList<ImportIssue> OutsideCore);
