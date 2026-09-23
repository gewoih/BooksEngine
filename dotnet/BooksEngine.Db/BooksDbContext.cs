using BooksEngine.Db.Entities;
using Microsoft.EntityFrameworkCore;

namespace BooksEngine.Db;

/// <summary>
/// Владелец схемы БД. Каталог наполняет Python (`booksengine load-db`) через COPY в эти таблицы;
/// пользовательские таблицы (app_users, ratings, shelves) пишет API.
/// </summary>
public class BooksDbContext(DbContextOptions<BooksDbContext> options) : DbContext(options)
{
    public static class EntityTypes
    {
        public const string Work = "work";
        public const string Edition = "edition";
        public const string Author = "author";
        public static readonly string[] All = [Work, Edition, Author];
    }

    public DbSet<Source> Sources => Set<Source>();
    public DbSet<ExternalId> ExternalIds => Set<ExternalId>();
    public DbSet<SourceLoad> SourceLoads => Set<SourceLoad>();
    public DbSet<Work> Works => Set<Work>();
    public DbSet<Edition> Editions => Set<Edition>();
    public DbSet<Author> Authors => Set<Author>();
    public DbSet<WorkAuthor> WorkAuthors => Set<WorkAuthor>();
    public DbSet<Genre> Genres => Set<Genre>();
    public DbSet<WorkGenre> WorkGenres => Set<WorkGenre>();
    public DbSet<AppUser> AppUsers => Set<AppUser>();
    public DbSet<Rating> Ratings => Set<Rating>();
    public DbSet<ShelfEntry> Shelves => Set<ShelfEntry>();
    public DbSet<WorkEmbedding> WorkEmbeddings => Set<WorkEmbedding>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        b.HasPostgresExtension("vector");
        b.HasPostgresExtension("pg_trgm");

        b.Entity<Source>(e =>
        {
            e.HasIndex(x => x.Code).IsUnique();
            e.HasData(new Source { Id = 1, Code = "goodreads" });
        });

        b.Entity<ExternalId>(e =>
        {
            e.HasKey(x => new { x.SourceId, x.EntityType, x.Value });
            e.Property(x => x.Value).HasColumnName("external_id");
            e.HasIndex(x => new { x.EntityType, x.InternalId });
            e.HasOne<Source>().WithMany().HasForeignKey(x => x.SourceId).OnDelete(DeleteBehavior.Restrict);
            e.ToTable(t => t.HasCheckConstraint("ck_external_ids_entity_type",
                $"entity_type IN ({string.Join(", ", EntityTypes.All.Select(s => $"'{s}'"))})"));
        });

        b.Entity<SourceLoad>(e =>
        {
            e.Property(x => x.Counts).HasColumnType("jsonb");
            e.Property(x => x.LoadedAt).HasDefaultValueSql("now()");
            e.HasOne<Source>().WithMany().HasForeignKey(x => x.SourceId).OnDelete(DeleteBehavior.Restrict);
        });

        b.Entity<Work>(e =>
        {
            e.HasOne(x => x.BestEdition).WithMany().HasForeignKey(x => x.BestEditionId)
                .OnDelete(DeleteBehavior.SetNull);
            // Поиск по названию с опечатками и неполным вводом (pg_trgm).
            e.HasIndex(x => x.Title).HasMethod("gin").HasOperators("gin_trgm_ops");
            e.HasIndex(x => x.OriginalTitle).HasMethod("gin").HasOperators("gin_trgm_ops");
        });

        b.Entity<Edition>(e =>
        {
            e.HasOne(x => x.Work).WithMany(x => x.Editions).HasForeignKey(x => x.WorkId)
                .OnDelete(DeleteBehavior.Cascade);
            e.HasIndex(x => x.Isbn);
            e.HasIndex(x => x.Isbn13);
            e.HasIndex(x => x.Title).HasMethod("gin").HasOperators("gin_trgm_ops");
        });

        b.Entity<Author>(e =>
        {
            e.HasIndex(x => x.Name).HasMethod("gin").HasOperators("gin_trgm_ops");
        });

        b.Entity<WorkAuthor>(e =>
        {
            e.HasKey(x => new { x.WorkId, x.AuthorId });
            e.HasIndex(x => x.AuthorId);
            e.HasOne<Work>().WithMany().HasForeignKey(x => x.WorkId).OnDelete(DeleteBehavior.Cascade);
            e.HasOne<Author>().WithMany().HasForeignKey(x => x.AuthorId).OnDelete(DeleteBehavior.Cascade);
        });

        b.Entity<Genre>(e => e.HasIndex(x => x.Name).IsUnique());

        b.Entity<WorkGenre>(e =>
        {
            e.HasKey(x => new { x.WorkId, x.GenreId });
            e.HasIndex(x => x.GenreId);
            e.HasOne<Work>().WithMany().HasForeignKey(x => x.WorkId).OnDelete(DeleteBehavior.Cascade);
            e.HasOne<Genre>().WithMany().HasForeignKey(x => x.GenreId).OnDelete(DeleteBehavior.Restrict);
        });

        b.Entity<AppUser>(e =>
        {
            e.HasIndex(x => x.Name).IsUnique();
            e.Property(x => x.CreatedAt).HasDefaultValueSql("now()");
        });

        // Данные пользователей ссылаются на каталог с Restrict: перезагрузка каталога,
        // которая удалила бы оценённое произведение, падает, а не стирает оценки молча.
        b.Entity<Rating>(e =>
        {
            e.HasKey(x => new { x.UserId, x.WorkId });
            e.HasIndex(x => x.WorkId);
            e.Property(x => x.ScaleMax).HasDefaultValue((short)10);
            e.Property(x => x.CreatedAt).HasDefaultValueSql("now()");
            e.Property(x => x.UpdatedAt).HasDefaultValueSql("now()");
            e.HasOne<AppUser>().WithMany().HasForeignKey(x => x.UserId).OnDelete(DeleteBehavior.Cascade);
            e.HasOne<Work>().WithMany().HasForeignKey(x => x.WorkId).OnDelete(DeleteBehavior.Restrict);
            e.ToTable(t => t.HasCheckConstraint("ck_ratings_value", "value BETWEEN 1 AND scale_max"));
        });

        b.Entity<ShelfEntry>(e =>
        {
            e.ToTable("shelves", t => t.HasCheckConstraint("ck_shelves_status",
                $"status IN ({string.Join(", ", ShelfStatus.All.Select(s => $"'{s}'"))})"));
            e.HasKey(x => new { x.UserId, x.WorkId });
            e.HasIndex(x => x.WorkId);
            e.Property(x => x.AddedAt).HasDefaultValueSql("now()");
            e.Property(x => x.UpdatedAt).HasDefaultValueSql("now()");
            e.HasOne<AppUser>().WithMany().HasForeignKey(x => x.UserId).OnDelete(DeleteBehavior.Cascade);
            e.HasOne<Work>().WithMany().HasForeignKey(x => x.WorkId).OnDelete(DeleteBehavior.Restrict);
        });

        b.Entity<WorkEmbedding>(e =>
        {
            e.HasKey(x => new { x.WorkId, x.ModelVersion });
            e.Property(x => x.Embedding).HasColumnType("vector");
            e.HasOne<Work>().WithMany().HasForeignKey(x => x.WorkId).OnDelete(DeleteBehavior.Cascade);
        });
    }
}
