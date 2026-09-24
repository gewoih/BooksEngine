using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace BooksEngine.Api.Import;

public enum CsvFormat { Ours, Goodreads }

/// <summary>Строка файла: ExternalId — Goodreads work_id (наш формат) или book_id издания (экспорт Goodreads).</summary>
public sealed record CsvRow(int Line, string Label, string ExternalId, int? Rating, bool Dnf, string? Error);

/// <summary>
/// Наш формат — как profiles/my_ratings.csv: goodreads_work_id, rating 1–5, status (dnf без оценки = 1), title.
/// Экспорт Goodreads (My Books → Export): Book Id, Title, My Rating (0 — без оценки).
/// </summary>
public static class RatingsCsv
{
    public static (CsvFormat Format, List<CsvRow> Rows) Parse(Stream stream)
    {
        using var reader = new StreamReader(stream, detectEncodingFromByteOrderMarks: true);
        using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            MissingFieldFound = null, BadDataFound = null, TrimOptions = TrimOptions.Trim,
        });
        if (!csv.Read() || !csv.ReadHeader()) throw new FormatException("файл пустой");
        var header = csv.HeaderRecord!;
        var format = header.Contains("goodreads_work_id") && header.Contains("rating") ? CsvFormat.Ours
            : header.Contains("Book Id") && header.Contains("My Rating") ? CsvFormat.Goodreads
            : throw new FormatException("нужен наш формат (goodreads_work_id, rating, status) или экспорт Goodreads (Book Id, My Rating)");

        var rows = new List<CsvRow>();
        while (csv.Read())
        {
            var line = csv.Parser.Row;
            if (format == CsvFormat.Ours)
            {
                var id = csv.GetField("goodreads_work_id") ?? "";
                var label = csv.GetField("title") is { Length: > 0 } t ? t : id;
                var dnf = csv.GetField("status") == "dnf";
                var raw = csv.GetField("rating") ?? "";
                int? rating = raw.Length == 0 && dnf ? 1 : int.TryParse(raw, out var v) ? v : null;
                var error = rating is >= 1 and <= 5 ? null : $"оценка «{raw}» — нужно целое 1–5";
                rows.Add(new CsvRow(line, label, id, rating, dnf, error));
            }
            else
            {
                var raw = csv.GetField("My Rating") ?? "";
                int? rating = int.TryParse(raw, out var v) ? v : null;
                var error = rating is >= 0 and <= 5 ? null : $"оценка «{raw}» — нужно целое 0–5";
                rows.Add(new CsvRow(line, csv.GetField("Title") ?? "", csv.GetField("Book Id") ?? "", rating, false, error));
            }
        }
        return (format, rows);
    }
}
