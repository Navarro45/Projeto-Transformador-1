import csv
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np

from shared.metrics import save_metrics_bundle


METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def load_metrics(results_dir):

    path = os.path.join(
        results_dir,
        "metrics",
        "metrics.json"
    )

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


def summarize_metric_runs(run_records):

    grouped = {}

    for record in run_records:

        key = record["name"]

        grouped.setdefault(
            key,
            []
        ).append(record)

    summary = {}

    for name, records in grouped.items():

        metrics_summary = {}

        for metric in METRIC_KEYS:

            values = [
                item["metrics"][metric]
                for item in records
                if metric in item["metrics"]
            ]

            if not values:

                continue

            metrics_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }

        summary[name] = {
            "runs": len(records),
            "metrics": metrics_summary,
            "results_dirs": [
                item["results_dir"]
                for item in records
            ],
        }

    return summary


def _load_prediction_bundle(results_dir):

    predictions_dir = os.path.join(
        results_dir,
        "predictions"
    )

    return {
        "labels": np.load(
            os.path.join(
                predictions_dir,
                "labels.npy"
            )
        ),
        "predictions": np.load(
            os.path.join(
                predictions_dir,
                "predictions.npy"
            )
        ),
        "probabilities": np.load(
            os.path.join(
                predictions_dir,
                "probabilities.npy"
            )
        ),
    }


def _majority_vote(prediction_matrix, num_classes):

    votes = []

    for row in prediction_matrix:

        counts = np.bincount(
            row.astype(int),
            minlength=num_classes
        )

        votes.append(int(np.argmax(counts)))

    return np.array(votes)


def _class_name(class_names, label):

    label = int(label)

    if 0 <= label < len(class_names):

        return class_names[label]

    return str(label)


def _write_strategy_decisions(
    output_dir,
    labels,
    predictions,
    class_names,
):

    path = os.path.join(
        output_dir,
        "decisions.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sample_index",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "correct",
        ])

        for sample_index, true_label in enumerate(labels):

            predicted_label = int(
                predictions[sample_index]
            )

            writer.writerow([
                sample_index,
                int(true_label),
                _class_name(class_names, true_label),
                predicted_label,
                _class_name(class_names, predicted_label),
                bool(predicted_label == int(true_label)),
            ])


def _write_comparison_decisions(
    run_output_dir,
    labels,
    strategy_predictions,
    model_predictions,
    model_probabilities,
    model_names,
    class_names,
):

    path = os.path.join(
        run_output_dir,
        "fusion_decision_comparison.csv"
    )

    headers = [
        "sample_index",
        "true_label",
        "true_class",
    ]

    for strategy_name in strategy_predictions.keys():

        headers.extend([
            f"{strategy_name}_label",
            f"{strategy_name}_class",
            f"{strategy_name}_correct",
        ])

    headers.append("strategies_agree")

    for model_name in model_names:

        headers.extend([
            f"{model_name}_label",
            f"{model_name}_class",
            f"{model_name}_confidence",
        ])

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(headers)

        for sample_index, true_label in enumerate(labels):

            row = [
                sample_index,
                int(true_label),
                _class_name(class_names, true_label),
            ]

            strategy_labels = []

            for predictions in strategy_predictions.values():

                predicted_label = int(
                    predictions[sample_index]
                )

                strategy_labels.append(predicted_label)

                row.extend([
                    predicted_label,
                    _class_name(class_names, predicted_label),
                    bool(predicted_label == int(true_label)),
                ])

            row.append(
                len(set(strategy_labels)) == 1
            )

            for model_index, model_name in enumerate(model_names):

                predicted_label = int(
                    model_predictions[model_index][sample_index]
                )

                confidence = float(
                    model_probabilities[model_index][sample_index][predicted_label]
                )

                row.extend([
                    predicted_label,
                    _class_name(class_names, predicted_label),
                    confidence,
                ])

            writer.writerow(row)


def evaluate_ensemble(
    run_records,
    class_names,
    base_output_dir="Resultados",
):

    if len(run_records) < 2:

        print(
            "Fusao ignorada: sao necessarios "
            "pelo menos dois resultados."
        )

        return []

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_%f"
    )

    ensemble_base_dir = os.path.join(
        base_output_dir,
        "ensemble",
        timestamp
    )

    os.makedirs(
        ensemble_base_dir,
        exist_ok=True
    )

    records_by_run = {}

    for record in run_records:

        records_by_run.setdefault(
            record["run_index"],
            []
        ).append(record)

    ensemble_records = []

    for run_index, records in records_by_run.items():

        if len(records) < 2:

            continue

        bundles = [
            _load_prediction_bundle(
                record["results_dir"]
            )
            for record in records
        ]

        labels = bundles[0]["labels"]

        for bundle in bundles[1:]:

            if not np.array_equal(
                labels,
                bundle["labels"]
            ):

                raise ValueError(
                    "Labels incompativeis entre modelos "
                    f"na run {run_index}."
                )

        probabilities = [
            bundle["probabilities"]
            for bundle in bundles
        ]

        predictions = [
            bundle["predictions"]
            for bundle in bundles
        ]

        num_classes = probabilities[0].shape[1]

        model_names = [
            record["name"]
            for record in records
        ]

        strategies = {
            "majority_vote": _majority_vote(
                np.stack(predictions, axis=1),
                num_classes
            ),
            "mean_probability": np.argmax(
                np.mean(
                    np.stack(probabilities, axis=0),
                    axis=0
                ),
                axis=1
            ),
        }

        run_output_dir = os.path.join(
            ensemble_base_dir,
            f"run_{run_index}"
        )

        for strategy_name, ensemble_pred in strategies.items():

            output_dir = os.path.join(
                run_output_dir,
                strategy_name
            )

            metrics_dir = os.path.join(
                output_dir,
                "metrics"
            )

            plots_dir = os.path.join(
                output_dir,
                "plots"
            )

            metrics = save_metrics_bundle(
                labels,
                ensemble_pred,
                class_names,
                metrics_dir,
                plots_dir,
                label=strategy_name
            )

            np.save(
                os.path.join(
                    output_dir,
                    "predictions.npy"
                ),
                ensemble_pred
            )

            _write_strategy_decisions(
                output_dir,
                labels,
                ensemble_pred,
                class_names
            )

            metadata = {
                "strategy": strategy_name,
                "run_index": run_index,
                "models": model_names,
                "source_results_dirs": [
                    record["results_dir"]
                    for record in records
                ],
            }

            with open(
                os.path.join(
                    output_dir,
                    "metadata.json"
                ),
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4
                )

            ensemble_records.append(
                {
                    "name": strategy_name,
                    "run_index": run_index,
                    "results_dir": output_dir,
                    "metrics": metrics,
                }
            )

        _write_comparison_decisions(
            run_output_dir,
            labels,
            strategies,
            predictions,
            probabilities,
            model_names,
            class_names
        )

    summary = summarize_metric_runs(
        ensemble_records
    )

    with open(
        os.path.join(
            ensemble_base_dir,
            "summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "summary": summary,
                "runs": ensemble_records,
            },
            f,
            indent=4
        )

    print(
        f"Resumo da fusao salvo em: {ensemble_base_dir}"
    )

    missing = [
        run_index
        for run_index, records in records_by_run.items()
        if len(records) < 2
    ]

    if missing:

        counts = Counter(missing)

        print(
            "Algumas runs foram ignoradas na fusao: "
            f"{list(counts.keys())}"
        )

    return ensemble_records
