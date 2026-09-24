using BooksEngine.Db;
using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace BooksEngine.Api.Tests;

/// <summary>
/// Отдельная БД booksengine_api_test: схема — из EF-миграций, каталог — 6 книг, модель — крошечная.
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
        await SeedModelAsync();
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
    // 1777 — тень Emma (вне ядра, слита в 1004 через work_merges; её издание 2777 — в каталоге).
    public Task SeedCatalogAsync() => ExecAsync("""
        INSERT INTO authors (id, name) VALUES (1, 'Frank Herbert'), (2, 'Stanisław Lem'), (3, 'Jane Austen'), (4, 'J.R.R. Tolkien');
        INSERT INTO works (id, title, publication_year, is_collection, in_cf, cf_ratings, description) VALUES
          (1001, 'Dune', 1965, false, true, 5000, 'spice'),
          (1002, 'Dune Messiah', 1969, false, true, 3000, NULL),
          (1003, 'Solaris', 1961, false, true, 800, NULL),
          (1004, 'Emma', 1815, false, true, 4000, NULL),
          (1005, 'Rare Book', 2001, false, false, 3, NULL),
          (1006, 'The Hobbit', 1937, false, true, 9000, NULL),
          (1777, 'Emma', 1815, false, false, 2, NULL);
        INSERT INTO editions (id, work_id, title) VALUES
          (2001, 1001, 'Dune'), (2002, 1003, 'Солярис'), (2003, 1004, 'Emma'), (2004, 1005, 'Rare Book'), (2005, 1006, 'Хоббит'), (2777, 1777, 'Эмма');
        INSERT INTO work_authors (work_id, author_id, role, position) VALUES
          (1001, 1, NULL, 0), (1002, 1, NULL, 0), (1003, 2, NULL, 0), (1004, 3, NULL, 0), (1005, 3, NULL, 0), (1006, 4, NULL, 0), (1777, 3, NULL, 0);
        INSERT INTO external_ids (source_id, entity_type, external_id, internal_id)
          SELECT 1, 'work', (id - 1000)::text, id FROM works
          UNION ALL SELECT 1, 'edition', (id - 2000)::text, id FROM editions;
        INSERT INTO work_covers (work_id, image_url) VALUES (1001, 'https://images.gr-assets.com/books/1m/1.jpg');
        INSERT INTO work_merges (shadow_external_id, main_work_id) VALUES ('777', 1004);
        """);

    // Модель на 5 книгах ядра (2 фактора), все в EASE. Числа не «правильные» — проверяется поведение эндпоинтов;
    // математику проверяют RecommenderTests (на руках) и GoldenTests (настоящая модель).
    public Task SeedModelAsync() => ExecAsync("""
        DELETE FROM model_meta; DELETE FROM work_embeddings; DELETE FROM ease_weights; DELETE FROM work_exclusions;
        INSERT INTO model_meta (id, fingerprint, params) VALUES (1, 'test', '{"als_weight":0.5,"ease_input":[-2,-1,0,1,2],"dnf_input":0.0,
          "als_alpha":1.0,"als_regularization":0.1,"als_neg_rule":"le2","als_neg_weight":0.0,
          "chance_coef":[-0.9,-0.46,0.85],"chance_prior":3.0,"chance_p0":0.72,
          "max_because":3,"min_of_leader":0.25,"despite_of_leader":0.5}');
        INSERT INTO work_embeddings (work_id, col, ease_pos, embedding) VALUES
          (1001, 0, 0, '[1,0]'), (1002, 1, 1, '[0.9,0.1]'), (1003, 2, 2, '[0.2,1]'), (1004, 3, 3, '[0,1]'), (1006, 4, 4, '[0.7,0.7]');
        INSERT INTO ease_weights (from_pos, to_pos, weight) VALUES
          (0, 1, 0.9), (0, 4, 0.6), (2, 3, 0.5), (3, 2, 0.4), (4, 0, 0.3);
        INSERT INTO work_exclusions (rated_work_id, excluded_work_id, reason) VALUES (1001, 1002, 'series'), (1002, 1001, 'series');
        """);

    public Task ClearModelAsync() => ExecAsync("DELETE FROM model_meta;");
}

[CollectionDefinition("db")]
public sealed class DbCollection : ICollectionFixture<TestDb>;
