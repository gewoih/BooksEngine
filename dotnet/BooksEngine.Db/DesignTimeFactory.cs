using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace BooksEngine.Db;

/// <summary>Для `dotnet ef`: строка подключения из POSTGRES_* (переменные окружения или .env в корне репозитория).</summary>
public class DesignTimeFactory : IDesignTimeDbContextFactory<BooksDbContext>
{
    public BooksDbContext CreateDbContext(string[] args)
    {
        var options = new DbContextOptionsBuilder<BooksDbContext>();
        Configure(options, ConnectionString.FromEnvironment());
        return new BooksDbContext(options.Options);
    }

    public static void Configure(DbContextOptionsBuilder options, string connectionString) =>
        options.UseNpgsql(connectionString, o => o.UseVector()).UseSnakeCaseNamingConvention();
}

public static class ConnectionString
{
    public static string FromEnvironment()
    {
        var env = ReadDotEnv();
        string Get(string key, string fallback) =>
            Environment.GetEnvironmentVariable(key) ?? env.GetValueOrDefault(key) ?? fallback;

        return $"Host={Get("POSTGRES_HOST", "localhost")};Port={Get("POSTGRES_PORT", "5432")};" +
               $"Database={Get("POSTGRES_DB", "booksengine")};Username={Get("POSTGRES_USER", "booksengine")};" +
               $"Password={Get("POSTGRES_PASSWORD", "booksengine")}";
    }

    private static Dictionary<string, string> ReadDotEnv()
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
        {
            var path = Path.Combine(dir.FullName, ".env");
            if (!File.Exists(path)) continue;
            return File.ReadAllLines(path)
                .Select(l => l.Trim())
                .Where(l => l.Length > 0 && !l.StartsWith('#') && l.Contains('='))
                .Select(l => l.Split('=', 2))
                .ToDictionary(p => p[0].Trim(), p => p[1].Trim());
        }
        return [];
    }
}
