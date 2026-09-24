using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BooksEngine.Db.Migrations
{
    /// <inheritdoc />
    public partial class ModelExport : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // старые строки без col; модель всё равно перевыгружается целиком (`booksengine export-model`)
            migrationBuilder.Sql("DELETE FROM work_embeddings;");

            migrationBuilder.DropPrimaryKey(
                name: "pk_work_embeddings",
                table: "work_embeddings");

            migrationBuilder.DropColumn(
                name: "model_version",
                table: "work_embeddings");

            migrationBuilder.AddColumn<int>(
                name: "col",
                table: "work_embeddings",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "ease_pos",
                table: "work_embeddings",
                type: "integer",
                nullable: true);

            migrationBuilder.AddPrimaryKey(
                name: "pk_work_embeddings",
                table: "work_embeddings",
                column: "work_id");

            migrationBuilder.CreateTable(
                name: "ease_weights",
                columns: table => new
                {
                    from_pos = table.Column<int>(type: "integer", nullable: false),
                    to_pos = table.Column<int>(type: "integer", nullable: false),
                    weight = table.Column<float>(type: "real", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_ease_weights", x => new { x.from_pos, x.to_pos });
                });

            migrationBuilder.CreateTable(
                name: "golden_inputs",
                columns: table => new
                {
                    profile = table.Column<string>(type: "text", nullable: false),
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    rating = table.Column<double>(type: "double precision", nullable: false),
                    dnf = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_golden_inputs", x => new { x.profile, x.work_id });
                });

            migrationBuilder.CreateTable(
                name: "golden_recommendations",
                columns: table => new
                {
                    profile = table.Column<string>(type: "text", nullable: false),
                    rank = table.Column<int>(type: "integer", nullable: false),
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    score = table.Column<double>(type: "double precision", nullable: false),
                    chance = table.Column<int>(type: "integer", nullable: false),
                    because = table.Column<long[]>(type: "bigint[]", nullable: false),
                    despite = table.Column<long>(type: "bigint", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_golden_recommendations", x => new { x.profile, x.rank });
                });

            migrationBuilder.CreateTable(
                name: "model_meta",
                columns: table => new
                {
                    id = table.Column<int>(type: "integer", nullable: false),
                    fingerprint = table.Column<string>(type: "text", nullable: false),
                    @params = table.Column<string>(name: "params", type: "jsonb", nullable: false),
                    exported_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()")
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_model_meta", x => x.id);
                    table.CheckConstraint("ck_model_meta_single", "id = 1");
                });

            migrationBuilder.CreateTable(
                name: "work_covers",
                columns: table => new
                {
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    image_url = table.Column<string>(type: "text", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_covers", x => x.work_id);
                    table.ForeignKey(
                        name: "fk_work_covers_works_work_id",
                        column: x => x.work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "work_exclusions",
                columns: table => new
                {
                    rated_work_id = table.Column<long>(type: "bigint", nullable: false),
                    excluded_work_id = table.Column<long>(type: "bigint", nullable: false),
                    reason = table.Column<string>(type: "text", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_exclusions", x => new { x.rated_work_id, x.excluded_work_id });
                    table.CheckConstraint("ck_work_exclusions_reason", "reason IN ('series', 'rated')");
                    table.ForeignKey(
                        name: "fk_work_exclusions_works_excluded_work_id",
                        column: x => x.excluded_work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_work_exclusions_works_rated_work_id",
                        column: x => x.rated_work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "work_merges",
                columns: table => new
                {
                    shadow_external_id = table.Column<string>(type: "text", nullable: false),
                    main_work_id = table.Column<long>(type: "bigint", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_merges", x => x.shadow_external_id);
                    table.ForeignKey(
                        name: "fk_work_merges_works_main_work_id",
                        column: x => x.main_work_id,
                        principalTable: "works",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "ix_work_embeddings_col",
                table: "work_embeddings",
                column: "col",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_work_embeddings_ease_pos",
                table: "work_embeddings",
                column: "ease_pos",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_work_exclusions_excluded_work_id",
                table: "work_exclusions",
                column: "excluded_work_id");

            migrationBuilder.CreateIndex(
                name: "ix_work_merges_main_work_id",
                table: "work_merges",
                column: "main_work_id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "ease_weights");

            migrationBuilder.DropTable(
                name: "golden_inputs");

            migrationBuilder.DropTable(
                name: "golden_recommendations");

            migrationBuilder.DropTable(
                name: "model_meta");

            migrationBuilder.DropTable(
                name: "work_covers");

            migrationBuilder.DropTable(
                name: "work_exclusions");

            migrationBuilder.DropTable(
                name: "work_merges");

            migrationBuilder.DropPrimaryKey(
                name: "pk_work_embeddings",
                table: "work_embeddings");

            migrationBuilder.DropIndex(
                name: "ix_work_embeddings_col",
                table: "work_embeddings");

            migrationBuilder.DropIndex(
                name: "ix_work_embeddings_ease_pos",
                table: "work_embeddings");

            migrationBuilder.DropColumn(
                name: "col",
                table: "work_embeddings");

            migrationBuilder.DropColumn(
                name: "ease_pos",
                table: "work_embeddings");

            migrationBuilder.AddColumn<string>(
                name: "model_version",
                table: "work_embeddings",
                type: "text",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddPrimaryKey(
                name: "pk_work_embeddings",
                table: "work_embeddings",
                columns: new[] { "work_id", "model_version" });
        }
    }
}
