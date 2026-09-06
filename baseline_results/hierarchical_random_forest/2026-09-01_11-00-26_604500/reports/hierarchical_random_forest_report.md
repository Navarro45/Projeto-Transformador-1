# hierarchical_random_forest

## Configuracao

- Amostras validas: 120
- Features espectrais: {'n_estimators': 8, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True, 'class_weight': 'balanced_subsample', 'max_samples': None, 'random_state': 42, 'n_jobs': 1, 'prediction_mode': 'global_probability'}
- Cache de features: dataset\star_supervised\cache\spectral_features_d4af38ea59606b58.npz
- Split: {'train': 79, 'test': 22, 'validation': 19}
- Parents: 11
- Leaf labels: 39

## Nota metodologica

O Random Forest e treinado com o criterio interno do scikit-learn. A hierarchical loss nao substitui backpropagation nem gradient descent; ela e usada para avaliacao e comparacao dos caminhos hierarquicos.

## Distribuicao

| spectral_parent_label | spectral_leaf_label | count | percentage | is_rare |
| --- | --- | --- | --- | --- |
| F | F9 | 15 | 12.5 | False |
| F | F5 | 13 | 10.833333333333334 | False |
| M | M5 | 7 | 5.833333333333333 | False |
| K | K3 | 6 | 5.0 | False |
| G | G2 | 5 | 4.166666666666666 | True |
| A | A0 | 5 | 4.166666666666666 | True |
| F | F2 | 5 | 4.166666666666666 | True |
| K | K5 | 5 | 4.166666666666666 | True |
| M | M2 | 5 | 4.166666666666666 | True |
| F | F3 | 5 | 4.166666666666666 | True |
| G | G0 | 4 | 3.3333333333333335 | True |
| F | F8 | 3 | 2.5 | True |
| M | M1 | 3 | 2.5 | True |
| K | K1 | 3 | 2.5 | True |
| M | M0 | 3 | 2.5 | True |
| WD | WD | 3 | 2.5 | True |
| M | M4 | 3 | 2.5 | True |
| A | A8 | 3 | 2.5 | True |
| L | L5.5 | 2 | 1.6666666666666667 | True |
| K | K7 | 2 | 1.6666666666666667 | True |
| K | K0 | 2 | 1.6666666666666667 | True |
| A | A6 | 1 | 0.8333333333333334 | True |
| CARBON | CARBON | 1 | 0.8333333333333334 | True |
| F | F0 | 1 | 0.8333333333333334 | True |
| L | L1 | 1 | 0.8333333333333334 | True |
| M | M7 | 1 | 0.8333333333333334 | True |
| G | G5 | 1 | 0.8333333333333334 | True |
| A | A9 | 1 | 0.8333333333333334 | True |
| A | A4 | 1 | 0.8333333333333334 | True |
| G | G1 | 1 | 0.8333333333333334 | True |
| M | M4.5 | 1 | 0.8333333333333334 | True |
| K | K4 | 1 | 0.8333333333333334 | True |
| T | T2 | 1 | 0.8333333333333334 | True |
| M | M3 | 1 | 0.8333333333333334 | True |
| G | G4 | 1 | 0.8333333333333334 | True |
| OB | OB | 1 | 0.8333333333333334 | True |
| G | G8 | 1 | 0.8333333333333334 | True |
| L | L9 | 1 | 0.8333333333333334 | True |
| CV | CV | 1 | 0.8333333333333334 | True |

## Resultados

### hierarchical_random_forest

- Parent accuracy: 0.5909090909090909
- Leaf accuracy: 0.13636363636363635
- Leaf macro F1: 0.02849002849002849
- Hierarchical loss: 7.920632537865715
- Average hierarchical distance: 2.5
- Same-parent errors: 10
- Cross-parent errors: 9

### flat_random_forest

- Parent accuracy: 0.5909090909090909
- Leaf accuracy: 0.13636363636363635
- Leaf macro F1: 0.034188034188034185
- Hierarchical loss: 16.038363687073556
- Average hierarchical distance: 2.5
- Same-parent errors: 10
- Cross-parent errors: 9

## Tempos

{
    "feature_time_seconds": 0.11412060000293422,
    "split_time_seconds": 0.013553899996622931,
    "training_time_seconds": 0.08694310000282712,
    "evaluation_time_seconds": 0.9376532999958727
}

## Saidas

- Modelo: Resultados\hierarchical_random_forest\2026-09-01_11-00-26_604500\model\model.joblib
- Metricas: Resultados\hierarchical_random_forest\2026-09-01_11-00-26_604500\metrics\metrics.json
- Predicoes: Resultados\hierarchical_random_forest\2026-09-01_11-00-26_604500\predictions\test_predictions.csv
- Relatorio: Resultados\hierarchical_random_forest\2026-09-01_11-00-26_604500\reports\hierarchical_random_forest_report.md