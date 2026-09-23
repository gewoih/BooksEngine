using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;
using Pgvector;

#nullable disable

namespace BooksEngine.Db.Migrations
{
    /// <inheritdoc />
    public partial class InitialCatalog : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterDatabase()
                .Annotation("Npgsql:PostgresExtension:pg_trgm", ",,")
                .Annotation("Npgsql:PostgresExtension:vector", ",,");

            migrationBuilder.CreateTable(
                name: "app_users",
                columns: table => new
                {
                    id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    name = table.Column<string>(type: "text", nullable: false),
                    created_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_app_users", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "authors",
                columns: table => new
                {
                    id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    name = table.Column<string>(type: "text", nullable: true),
                    average_rating = table.Column<double>(type: "double precision", nullable: true),
                    ratings_count = table.Column<long>(type: "bigint", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_authors", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "genres",
                columns: table => new
                {
                    id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    name = table.Column<string>(type: "text", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_genres", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "sources",
                columns: table => new
                {
                    id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    code = table.Column<string>(type: "text", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_sources", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "external_ids",
                columns: table => new
                {
                    source_id = table.Column<int>(type: "integer", nullable: false),
                    entity_type = table.Column<string>(type: "text", nullable: false),
                    external_id = table.Column<string>(type: "text", nullable: false),
                    internal_id = table.Column<long>(type: "bigint", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_external_ids", x => new { x.source_id, x.entity_type, x.external_id });
                    table.CheckConstraint("ck_external_ids_entity_type", "entity_type IN ('work', 'edition', 'author')");
                    table.ForeignKey(
                        name: "fk_external_ids_sources_source_id",
                        column: x => x.source_id,
                        principalTable: "sources",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "source_loads",
                columns: table => new
                {
                    id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    source_id = table.Column<int>(type: "integer", nullable: false),
                    manifest_sha256 = table.Column<string>(type: "text", nullable: false),
                    counts = table.Column<string>(type: "jsonb", nullable: false),
                    loaded_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_source_loads", x => x.id);
                    table.ForeignKey(
                        name: "fk_source_loads_sources_source_id",
                        column: x => x.source_id,
                        principalTable: "sources",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "editions",
                columns: table => new
                {
                    id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    title = table.Column<string>(type: "text", nullable: true),
                    title_without_series = table.Column<string>(type: "text", nullable: true),
                    isbn = table.Column<string>(type: "text", nullable: true),
                    isbn13 = table.Column<string>(type: "text", nullable: true),
                    asin = table.Column<string>(type: "text", nullable: true),
                    kindle_asin = table.Column<string>(type: "text", nullable: true),
                    language_code = table.Column<string>(type: "text", nullable: true),
                    country_code = table.Column<string>(type: "text", nullable: true),
                    format = table.Column<string>(type: "text", nullable: true),
                    is_ebook = table.Column<bool>(type: "boolean", nullable: true),
                    num_pages = table.Column<int>(type: "integer", nullable: true),
                    publication_year = table.Column<int>(type: "integer", nullable: true),
                    publisher = table.Column<string>(type: "text", nullable: true),
                    ratings_count = table.Column<long>(type: "bigint", nullable: true),
                    average_rating = table.Column<double>(type: "double precision", nullable: true),
                    image_url = table.Column<string>(type: "text", nullable: true),
                    url = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_editions", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "works",
                columns: table => new
                {
                    id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    title = table.Column<string>(type: "text", nullable: false),
                    original_title = table.Column<string>(type: "text", nullable: true),
                    best_edition_id = table.Column<long>(type: "bigint", nullable: true),
                    publication_year = table.Column<int>(type: "integer", nullable: true),
                    language_code = table.Column<string>(type: "text", nullable: true),
                    description = table.Column<string>(type: "text", nullable: true),
                    is_collection = table.Column<bool>(type: "boolean", nullable: false),
                    in_cf = table.Column<bool>(type: "boolean", nullable: false),
                    cf_ratings = table.Column<int>(type: "integer", nullable: false),
                    cf_mean_rating = table.Column<double>(type: "double precision", nullable: true),
                    gr_ratings_count = table.Column<long>(type: "bigint", nullable: true),
                    gr_ratings_sum = table.Column<long>(type: "bigint", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_works", x => x.id);
                    table.ForeignKey(
                        name: "fk_works_editions_best_edition_id",
                        column: x => x.best_edition_id,
                        principalTable: "editions",
                        principalColumn: "id",
                        onDelete: ReferentialAction.SetNull);
                });

            migrationBuilder.CreateTable(
                name: "ratings",
                columns: table => new
                {
                    user_id = table.Column<long>(type: "bigint", nullable: false),
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    value = table.Column<short>(type: "smallint", nullable: false),
                    scale_max = table.Column<short>(type: "smallint", nullable: false, defaultValue: (short)10),
                    created_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()"),
                    updated_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_ratings", x => new { x.user_id, x.work_id });
                    table.CheckConstraint("ck_ratings_value", "value BETWEEN 1 AND scale_max");
                    table.ForeignKey(
                        name: "fk_ratings_app_users_user_id",
                        column: x => x.user_id,
                        principalTable: "app_users",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_ratings_works_work_id",
                        column: x => x.work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "shelves",
                columns: table => new
                {
                    user_id = table.Column<long>(type: "bigint", nullable: false),
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    status = table.Column<string>(type: "text", nullable: false),
                    added_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()"),
                    updated_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_shelves", x => new { x.user_id, x.work_id });
                    table.CheckConstraint("ck_shelves_status", "status IN ('to_read', 'reading', 'read', 'dnf')");
                    table.ForeignKey(
                        name: "fk_shelves_app_users_user_id",
                        column: x => x.user_id,
                        principalTable: "app_users",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_shelves_works_work_id",
                        column: x => x.work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "work_authors",
                columns: table => new
                {
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    author_id = table.Column<long>(type: "bigint", nullable: false),
                    role = table.Column<string>(type: "text", nullable: true),
                    position = table.Column<short>(type: "smallint", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_authors", x => new { x.work_id, x.author_id });
                    table.ForeignKey(
                        name: "fk_work_authors_authors_author_id",
                        column: x => x.author_id,
                        principalTable: "authors",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_work_authors_works_work_id",
                        column: x => x.work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "work_embeddings",
                columns: table => new
                {
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    model_version = table.Column<string>(type: "text", nullable: false),
                    embedding = table.Column<Vector>(type: "vector", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_embeddings", x => new { x.work_id, x.model_version });
                    table.ForeignKey(
                        name: "fk_work_embeddings_works_work_id",
                        column: x => x.work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "work_genres",
                columns: table => new
                {
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    genre_id = table.Column<int>(type: "integer", nullable: false),
                    votes = table.Column<int>(type: "integer", nullable: false),
                    share = table.Column<double>(type: "double precision", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_genres", x => new { x.work_id, x.genre_id });
                    table.ForeignKey(
                        name: "fk_work_genres_genres_genre_id",
                        column: x => x.genre_id,
                        principalTable: "genres",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_work_genres_works_work_id",
                        column: x => x.work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.InsertData(
                table: "sources",
                columns: new[] { "id", "code" },
                values: new object[] { 1, "goodreads" });

            migrationBuilder.CreateIndex(
                name: "ix_app_users_name",
                table: "app_users",
                column: "name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_authors_name",
                table: "authors",
                column: "name")
                .Annotation("Npgsql:IndexMethod", "gin")
                .Annotation("Npgsql:IndexOperators", new[] { "gin_trgm_ops" });

            migrationBuilder.CreateIndex(
                name: "ix_editions_isbn",
                table: "editions",
                column: "isbn");

            migrationBuilder.CreateIndex(
                name: "ix_editions_isbn13",
                table: "editions",
                column: "isbn13");

            migrationBuilder.CreateIndex(
                name: "ix_editions_title",
                table: "editions",
                column: "title")
                .Annotation("Npgsql:IndexMethod", "gin")
                .Annotation("Npgsql:IndexOperators", new[] { "gin_trgm_ops" });

            migrationBuilder.CreateIndex(
                name: "ix_editions_work_id",
                table: "editions",
                column: "work_id");

            migrationBuilder.CreateIndex(
                name: "ix_external_ids_entity_type_internal_id",
                table: "external_ids",
                columns: new[] { "entity_type", "internal_id" });

            migrationBuilder.CreateIndex(
                name: "ix_genres_name",
                table: "genres",
                column: "name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_ratings_work_id",
                table: "ratings",
                column: "work_id");

            migrationBuilder.CreateIndex(
                name: "ix_shelves_work_id",
                table: "shelves",
                column: "work_id");

            migrationBuilder.CreateIndex(
                name: "ix_source_loads_source_id",
                table: "source_loads",
                column: "source_id");

            migrationBuilder.CreateIndex(
                name: "ix_sources_code",
                table: "sources",
                column: "code",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_work_authors_author_id",
                table: "work_authors",
                column: "author_id");

            migrationBuilder.CreateIndex(
                name: "ix_work_genres_genre_id",
                table: "work_genres",
                column: "genre_id");

            migrationBuilder.CreateIndex(
                name: "ix_works_best_edition_id",
                table: "works",
                column: "best_edition_id");

            migrationBuilder.CreateIndex(
                name: "ix_works_original_title",
                table: "works",
                column: "original_title")
                .Annotation("Npgsql:IndexMethod", "gin")
                .Annotation("Npgsql:IndexOperators", new[] { "gin_trgm_ops" });

            migrationBuilder.CreateIndex(
                name: "ix_works_title",
                table: "works",
                column: "title")
                .Annotation("Npgsql:IndexMethod", "gin")
                .Annotation("Npgsql:IndexOperators", new[] { "gin_trgm_ops" });

            migrationBuilder.AddForeignKey(
                name: "fk_editions_works_work_id",
                table: "editions",
                column: "work_id",
                principalTable: "works",
                principalColumn: "id",
                onDelete: ReferentialAction.Cascade);

            // works ↔ editions ссылаются друг на друга. Отложенная проверка позволяет загрузчику
            // вставить best_edition_id сразу, а не отдельным UPDATE по всем произведениям
            // (UPDATE переиндексирует строку, включая триграммные GIN по названиям). EF-модель
            // DEFERRABLE не выражает, поэтому — SQL.
            migrationBuilder.Sql(
                "ALTER TABLE works ALTER CONSTRAINT fk_works_editions_best_edition_id DEFERRABLE INITIALLY IMMEDIATE;");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "fk_editions_works_work_id",
                table: "editions");

            migrationBuilder.DropTable(
                name: "external_ids");

            migrationBuilder.DropTable(
                name: "ratings");

            migrationBuilder.DropTable(
                name: "shelves");

            migrationBuilder.DropTable(
                name: "source_loads");

            migrationBuilder.DropTable(
                name: "work_authors");

            migrationBuilder.DropTable(
                name: "work_embeddings");

            migrationBuilder.DropTable(
                name: "work_genres");

            migrationBuilder.DropTable(
                name: "app_users");

            migrationBuilder.DropTable(
                name: "sources");

            migrationBuilder.DropTable(
                name: "authors");

            migrationBuilder.DropTable(
                name: "genres");

            migrationBuilder.DropTable(
                name: "works");

            migrationBuilder.DropTable(
                name: "editions");
        }
    }
}
