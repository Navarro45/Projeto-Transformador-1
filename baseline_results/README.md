# Baseline Results

Resultados separados dos modelos `flat_random_forest` e `hierarchical_random_forest`.

Esta pasta copia os artefatos leves para consulta e relatorio: metricas, metadados, plots, relatorios Markdown, `test_predictions.csv` e summaries.

Modelos `.joblib`, arrays `.npy/.npz` e outros binarios grandes foram deixados em `Resultados/` e registrados em `skipped_artifacts.csv` para evitar duplicacao pesada no repositorio.

Arquivos de controle:

- `manifest.csv`: arquivos copiados.
- `skipped_artifacts.csv`: artefatos ignorados e motivo.
- `summary.json`: resumo da exportacao.
