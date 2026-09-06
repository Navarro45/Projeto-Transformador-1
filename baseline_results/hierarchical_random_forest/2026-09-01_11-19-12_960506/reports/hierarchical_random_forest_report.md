# hierarchical_random_forest

## Configuracao

- Amostras validas: 500
- Features espectrais: {'n_estimators': 100, 'criterion': 'log_loss', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True, 'class_weight': 'balanced_subsample', 'max_samples': None, 'random_state': 42, 'n_jobs': -1, 'prediction_mode': 'global_probability'}
- Cache de features: dataset\star_supervised\cache\spectral_features_83da4231ccb8996a.npz
- Split: {'train': 356, 'test': 78, 'validation': 66}
- Parents: 13
- Leaf labels: 52

## Nota metodologica

O Random Forest e treinado com o criterio interno do scikit-learn. A hierarchical loss nao substitui backpropagation nem gradient descent; ela e usada para avaliacao e comparacao dos caminhos hierarquicos.

## Distribuicao

| spectral_parent_label | spectral_leaf_label | count | percentage | is_rare |
| --- | --- | --- | --- | --- |
| F | F9 | 55 | 11.0 | False |
| F | F5 | 48 | 9.6 | False |
| A | A0 | 47 | 9.4 | False |
| K | K5 | 33 | 6.6000000000000005 | False |
| K | K3 | 31 | 6.2 | False |
| G | G2 | 22 | 4.3999999999999995 | False |
| M | M4 | 21 | 4.2 | False |
| F | F3 | 19 | 3.8 | False |
| M | M1 | 16 | 3.2 | False |
| M | M5 | 16 | 3.2 | False |
| K | K1 | 15 | 3.0 | False |
| K | K0 | 14 | 2.8000000000000003 | False |
| M | M3 | 14 | 2.8000000000000003 | False |
| F | F2 | 11 | 2.1999999999999997 | False |
| F | F8 | 11 | 2.1999999999999997 | False |
| G | G0 | 10 | 2.0 | False |
| M | M2 | 10 | 2.0 | False |
| WD | WD | 10 | 2.0 | False |
| M | M0 | 8 | 1.6 | False |
| G | G5 | 6 | 1.2 | False |
| K | K7 | 6 | 1.2 | False |
| F | F0 | 6 | 1.2 | False |
| A | A4 | 6 | 1.2 | False |
| A | A8 | 5 | 1.0 | True |
| G | G8 | 5 | 1.0 | True |
| A | A9 | 4 | 0.8 | True |
| G | G9 | 4 | 0.8 | True |
| G | G4 | 4 | 0.8 | True |
| M | M4.5 | 4 | 0.8 | True |
| L | L5.5 | 3 | 0.6 | True |
| M | M6 | 3 | 0.6 | True |
| CARBON | CARBON | 2 | 0.4 | True |
| OB | OB | 2 | 0.4 | True |
| T | T2 | 2 | 0.4 | True |
| A | A6 | 2 | 0.4 | True |
| G | G1 | 2 | 0.4 | True |
| A | A2 | 2 | 0.4 | True |
| B | B5 | 2 | 0.4 | True |
| K | K4 | 2 | 0.4 | True |
| CV | CV | 2 | 0.4 | True |

_Mostrando 40 de 52 linhas._

## Resultados

### hierarchical_random_forest

- Parent accuracy: 0.6153846153846154
- Leaf accuracy: 0.38461538461538464
- Leaf macro F1: 0.1053609880742367
- Hierarchical loss: 2.6554575007227594
- Average hierarchical distance: 1.935897435897436
- Same-parent errors: 18
- Cross-parent errors: 30

### flat_random_forest

- Parent accuracy: 0.6410256410256411
- Leaf accuracy: 0.38461538461538464
- Leaf macro F1: 0.09210066249539933
- Hierarchical loss: 4.689905034419539
- Average hierarchical distance: 1.8974358974358974
- Same-parent errors: 20
- Cross-parent errors: 28

## Tempos

{
    "feature_time_seconds": 18.423143700005312,
    "split_time_seconds": 0.01578019999578828,
    "training_time_seconds": 1.6462612000032095,
    "evaluation_time_seconds": 2.218397199998435
}

## Saidas

- Modelo: Resultados\hierarchical_random_forest\2026-09-01_11-19-12_960506\model\model.joblib
- Metricas: Resultados\hierarchical_random_forest\2026-09-01_11-19-12_960506\metrics\metrics.json
- Predicoes: Resultados\hierarchical_random_forest\2026-09-01_11-19-12_960506\predictions\test_predictions.csv
- Relatorio: Resultados\hierarchical_random_forest\2026-09-01_11-19-12_960506\reports\hierarchical_random_forest_report.md