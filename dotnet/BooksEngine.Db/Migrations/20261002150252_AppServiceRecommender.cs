using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Pgvector;

#nullable disable

namespace BooksEngine.Db.Migrations
{
    /// <inheritdoc />
    public partial class AppServiceRecommender : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
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
                name: "work_embeddings");

            migrationBuilder.DropTable(
                name: "work_exclusions");

            migrationBuilder.InsertData(
                table: "sources",
                columns: new[] { "id", "code" },
                values: new object[] { 2, "amazon" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DeleteData(
                table: "sources",
                keyColumn: "id",
                keyValue: 2);

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
                    dnf = table.Column<bool>(type: "boolean", nullable: false),
                    rating = table.Column<double>(type: "double precision", nullable: false)
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
                    because = table.Column<long[]>(type: "bigint[]", nullable: false),
                    chance = table.Column<int>(type: "integer", nullable: false),
                    despite = table.Column<long>(type: "bigint", nullable: true),
                    score = table.Column<double>(type: "double precision", nullable: false),
                    work_id = table.Column<long>(type: "bigint", nullable: false)
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
                    exported_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false, defaultValueSql: "now()"),
                    fingerprint = table.Column<string>(type: "text", nullable: false),
                    @params = table.Column<string>(name: "params", type: "jsonb", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_model_meta", x => x.id);
                    table.CheckConstraint("ck_model_meta_single", "id = 1");
                });

            migrationBuilder.CreateTable(
                name: "work_embeddings",
                columns: table => new
                {
                    work_id = table.Column<long>(type: "bigint", nullable: false),
                    col = table.Column<int>(type: "integer", nullable: false),
                    ease_pos = table.Column<int>(type: "integer", nullable: true),
                    embedding = table.Column<Vector>(type: "vector", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_work_embeddings", x => x.work_id);
                    table.ForeignKey(
                        name: "fk_work_embeddings_works_work_id",
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
        }
    }
}
