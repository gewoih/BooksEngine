namespace BooksEngine.Api;

// Контракты API (JSON camelCase). Типы для фронта — из OpenAPI (`npm run gen:api` в web/).

public record UserDto(long Id, string Name);
public record CreateUserRequest(string Name);
public record SessionRequest(long UserId);

public record BookDto(long WorkId, string Title, string? Author, int? Year, string? CoverUrl, int CfRatings,
                      int? MyRating, bool MyDnf, bool InCore);
public record PageDto<T>(IReadOnlyList<T> Items, int Page, int PageSize, int Total);

/// <summary>Оценка 1–5 или «бросил» (Dnf = true, Value игнорируется): бросил = 1 + полка dnf.</summary>
public record RateRequest(int? Value, bool Dnf);
public record MyRatingDto(BookDto Book, DateTime UpdatedAt);

public record BookRef(long WorkId, string Title);
public record RecommendationDto(BookDto Book, int ChancePct, IReadOnlyList<BookRef> Because, BookRef? Despite);
public record RecommendationsDto(IReadOnlyList<RecommendationDto> Items, int UsedRatings);
public record WorkDetailDto(BookDto Book, string? LargeCoverUrl, string? Description, IReadOnlyList<string> Authors,
                            IReadOnlyList<string> Genres, int? ChancePct, bool InModel, IReadOnlyList<BookDto> Similar);
public record ReloadDto(string? Fingerprint);

public record ImportIssue(int Line, string Book, string Reason);
public record ImportResultDto(int Added, int Updated, int Skipped, IReadOnlyList<ImportIssue> NotFound,
                              IReadOnlyList<ImportIssue> Errors, IReadOnlyList<ImportIssue> OutsideCore);
