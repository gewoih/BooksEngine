using BooksEngine.Db;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace BooksEngine.Api.Tests;

/// <summary>
/// Отдельная БД booksengine_api_test: схема — из EF-миграций, каталог — 7 книг (выдачу подделывает FakeRecommender).
/// Нет Postgres — Available = false, тесты пропускаются (Assert.SkipUnless).
/// </summary>
public sealed class TestDb : IAsyncLifetime
{
    public const string Name = "booksengine_api_test";
    public bool Available { get; private set; }
    public string ConnectionString { get; private set; } = "";

    public async ValueTask InitializeAsync()
    {
        var admin = new NpgsqlConnectionStringBuilder(Db.ConnectionString.FromEnvironment()) { Timeout = 3 };
        try
        {
            await using var c = new NpgsqlConnection(admin.ConnectionString);
            await c.OpenAsync();
            await new NpgsqlCommand($"DROP DATABASE IF EXISTS {Name} WITH (FORCE)", c).ExecuteNonQueryAsync();
            await new NpgsqlCommand($"CREATE DATABASE {Name}", c).ExecuteNonQueryAsync();
        }
        catch (NpgsqlException) { return; }
        ConnectionString = new NpgsqlConnectionStringBuilder(admin.ConnectionString) { Database = Name }.ConnectionString;
        var options = new DbContextOptionsBuilder<BooksDbContext>();
        DesignTimeFactory.Configure(options, ConnectionString);
        await using (var db = new BooksDbContext(options.Options)) await db.Database.MigrateAsync();
        await SeedCatalogAsync();
        Available = true;
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    public async Task ExecAsync(string sql)
    {
        await using var c = new NpgsqlConnection(ConnectionString);
        await c.OpenAsync();
        await new NpgsqlCommand(sql, c).ExecuteNonQueryAsync();
    }

    // Внутренние id = Goodreads-id + 1000, чтобы тесты ловили путаницу id.
    // 1001 Dune, 1002 Dune Messiah (серия), 1003 Solaris, 1004 Emma, 1005 Rare (вне ядра), 1006 Hobbit,
    // 1777 — тень Emma (вне ядра, слита в 1004 через work_merges; её издание 2777 — в каталоге),
    // 1008 — новая книга Amazon (внешний id — ключ «автор|название»). По-русски — Дюна (Фрэнк Герберт) и Новая.
    public Task SeedCatalogAsync() => ExecAsync("""
        INSERT INTO authors (id, name) VALUES (1, 'Frank Herbert'), (2, 'Stanisław Lem'), (3, 'Jane Austen'), (4, 'J.R.R. Tolkien');
        INSERT INTO works (id, title, publication_year, is_collection, in_cf, cf_ratings, description) VALUES
          (1001, 'Dune', 1965, false, true, 5000, 'spice'),
          (1002, 'Dune Messiah', 1969, false, true, 3000, NULL),
          (1003, 'Solaris', 1961, false, true, 800, NULL),
          (1004, 'Emma', 1815, false, true, 4000, NULL),
          (1005, 'Rare Book', 2001, false, false, 3, NULL),
          (1006, 'The Hobbit', 1937, false, true, 9000, NULL),
          (1777, 'Emma', 1815, false, false, 2, NULL),
          (1008, 'New Book', 2019, false, true, 150, NULL);
        UPDATE works SET ru_title = 'Дюна' WHERE id = 1001;
        UPDATE works SET ru_title = 'Новая' WHERE id = 1008;
        UPDATE authors SET ru_name = 'Фрэнк Герберт' WHERE id = 1;
        INSERT INTO editions (id, work_id, title) VALUES
          (2001, 1001, 'Dune'), (2002, 1003, 'Солярис'), (2003, 1004, 'Emma'), (2004, 1005, 'Rare Book'), (2005, 1006, 'Хоббит'), (2777, 1777, 'Эмма');
        INSERT INTO work_authors (work_id, author_id, role, position) VALUES
          (1001, 1, NULL, 0), (1002, 1, NULL, 0), (1003, 2, NULL, 0), (1004, 3, NULL, 0), (1005, 3, NULL, 0), (1006, 4, NULL, 0), (1777, 3, NULL, 0);
        INSERT INTO external_ids (source_id, entity_type, external_id, internal_id)
          SELECT 1, 'work', (id - 1000)::text, id FROM works
          WHERE id <> 1008
          UNION ALL SELECT 1, 'edition', (id - 2000)::text, id FROM editions
          UNION ALL SELECT 2, 'work', 'author|new book', 1008;
        INSERT INTO work_covers (work_id, image_url) VALUES (1001, 'https://images.gr-assets.com/books/1m/1.jpg');
        INSERT INTO work_merges (shadow_external_id, main_work_id) VALUES ('777', 1004);
        """);

}

[CollectionDefinition("db")]
public sealed class DbCollection : ICollectionFixture<TestDb>;
