using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BooksEngine.Db.Migrations
{
    /// <inheritdoc />
    public partial class RatingsFivePoint : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropCheckConstraint(
                name: "ck_ratings_value",
                table: "ratings");

            migrationBuilder.DropColumn(
                name: "scale_max",
                table: "ratings");

            migrationBuilder.AddCheckConstraint(
                name: "ck_ratings_value",
                table: "ratings",
                sql: "value BETWEEN 1 AND 5");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropCheckConstraint(
                name: "ck_ratings_value",
                table: "ratings");

            migrationBuilder.AddColumn<short>(
                name: "scale_max",
                table: "ratings",
                type: "smallint",
                nullable: false,
                defaultValue: (short)10);

            migrationBuilder.AddCheckConstraint(
                name: "ck_ratings_value",
                table: "ratings",
                sql: "value BETWEEN 1 AND scale_max");
        }
    }
}
