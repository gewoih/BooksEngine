using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BooksEngine.Db.Migrations
{
    /// <inheritdoc />
    public partial class RussianNames : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "ru_title",
                table: "works",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ru_name",
                table: "authors",
                type: "text",
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "ix_works_ru_title",
                table: "works",
                column: "ru_title")
                .Annotation("Npgsql:IndexMethod", "gin")
                .Annotation("Npgsql:IndexOperators", new[] { "gin_trgm_ops" });

            migrationBuilder.CreateIndex(
                name: "ix_authors_ru_name",
                table: "authors",
                column: "ru_name")
                .Annotation("Npgsql:IndexMethod", "gin")
                .Annotation("Npgsql:IndexOperators", new[] { "gin_trgm_ops" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "ix_works_ru_title",
                table: "works");

            migrationBuilder.DropIndex(
                name: "ix_authors_ru_name",
                table: "authors");

            migrationBuilder.DropColumn(
                name: "ru_title",
                table: "works");

            migrationBuilder.DropColumn(
                name: "ru_name",
                table: "authors");
        }
    }
}
