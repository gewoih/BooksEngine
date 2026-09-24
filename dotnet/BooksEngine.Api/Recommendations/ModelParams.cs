namespace BooksEngine.Api.Recommendations;

/// <summary>Параметры из model_meta.params (JSON Python, snake_case) — `export_model.build`.</summary>
public sealed record ModelParams(
    double AlsWeight, double[] EaseInput, double DnfInput, double AlsAlpha, double AlsRegularization, string AlsNegRule,
    double AlsNegWeight, double[] ChanceCoef, double ChancePrior, double ChanceP0,
    int MaxBecause, double MinOfLeader, double DespiteOfLeader);
