# hierarchical_random_forest

## Configuracao

- Amostras validas: 9990
- Features espectrais: {'n_estimators': 300, 'criterion': 'log_loss', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True, 'class_weight': 'balanced_subsample', 'max_samples': None, 'random_state': 42, 'n_jobs': -1, 'prediction_mode': 'global_probability'}
- Cache de features: dataset\star_supervised\cache\spectral_features_80d00876b0042d72.npz
- Split: {'train': 7034, 'test': 1480, 'validation': 1476}
- Parents: 13
- Leaf labels: 70

## Nota metodologica

O Random Forest e treinado com o criterio interno do scikit-learn. A hierarchical loss nao substitui backpropagation nem gradient descent; ela e usada para avaliacao e comparacao dos caminhos hierarquicos.

## Distribuicao

| spectral_parent_label | spectral_leaf_label | count | percentage | is_rare |
| --- | --- | --- | --- | --- |
| A | A0 | 1120 | 11.21121121121121 | False |
| F | F9 | 1013 | 10.14014014014014 | False |
| F | F5 | 974 | 9.74974974974975 | False |
| K | K5 | 709 | 7.097097097097096 | False |
| K | K3 | 642 | 6.426426426426427 | False |
| F | F3 | 528 | 5.285285285285285 | False |
| K | K1 | 356 | 3.563563563563563 | False |
| M | M3 | 318 | 3.183183183183183 | False |
| G | G2 | 288 | 2.8828828828828827 | False |
| K | K0 | 281 | 2.8128128128128127 | False |
| M | M4 | 276 | 2.7627627627627627 | False |
| M | M2 | 262 | 2.6226226226226226 | False |
| M | M1 | 248 | 2.4824824824824825 | False |
| G | G0 | 243 | 2.4324324324324325 | False |
| F | F2 | 239 | 2.3923923923923924 | False |
| F | F0 | 216 | 2.1621621621621623 | False |
| M | M5 | 202 | 2.022022022022022 | False |
| WD | WD | 197 | 1.9719719719719717 | False |
| M | M0 | 175 | 1.7517517517517518 | False |
| G | G8 | 173 | 1.7317317317317318 | False |
| F | F8 | 167 | 1.6716716716716717 | False |
| K | K7 | 162 | 1.6216216216216217 | False |
| A | A4 | 133 | 1.3313313313313313 | False |
| G | G4 | 106 | 1.0610610610610611 | False |
| A | A1 | 71 | 0.7107107107107107 | False |
| A | A2 | 64 | 0.6406406406406406 | False |
| F | F6 | 62 | 0.6206206206206206 | False |
| M | M6 | 59 | 0.5905905905905906 | False |
| M | M4.5 | 57 | 0.5705705705705706 | False |
| CV | CV | 51 | 0.5105105105105104 | False |
| G | G5 | 50 | 0.5005005005005005 | False |
| A | A9 | 38 | 0.38038038038038036 | False |
| B | B5 | 34 | 0.34034034034034033 | False |
| K | K4 | 34 | 0.34034034034034033 | False |
| G | G1 | 34 | 0.34034034034034033 | False |
| G | G9 | 34 | 0.34034034034034033 | False |
| A | A6 | 29 | 0.2902902902902903 | False |
| B | B9 | 28 | 0.28028028028028024 | False |
| A | A8 | 27 | 0.2702702702702703 | False |
| A | A5 | 23 | 0.2302302302302302 | False |

_Mostrando 40 de 70 linhas._

## Resultados

### hierarchical_random_forest

- Parent accuracy: 0.7925675675675675
- Leaf accuracy: 0.5675675675675675
- Leaf macro F1: 0.19987659449586856
- Hierarchical loss: 1.2428032342357955
- Average hierarchical distance: 1.258108108108108
- Same-parent errors: 333
- Cross-parent errors: 307

### flat_random_forest

- Parent accuracy: 0.7885135135135135
- Leaf accuracy: 0.5581081081081081
- Leaf macro F1: 0.19751675615383646
- Hierarchical loss: 1.2774181252712353
- Average hierarchical distance: 1.2851351351351352
- Same-parent errors: 341
- Cross-parent errors: 313

## Tempos

{
    "feature_time_seconds": 525.578539400005,
    "split_time_seconds": 0.25134269999398384,
    "training_time_seconds": 55.14559370000643,
    "evaluation_time_seconds": 4.9745464000006905
}

## Saidas

- Modelo: Resultados\hierarchical_random_forest\2026-09-01_11-21-26_528425\model\model.joblib
- Metricas: Resultados\hierarchical_random_forest\2026-09-01_11-21-26_528425\metrics\metrics.json
- Predicoes: Resultados\hierarchical_random_forest\2026-09-01_11-21-26_528425\predictions\test_predictions.csv
- Relatorio: Resultados\hierarchical_random_forest\2026-09-01_11-21-26_528425\reports\hierarchical_random_forest_report.md