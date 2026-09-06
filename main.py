import argparse
import gc
import json
import os
import random

import torch

from legacy.data.data_module import DataModule

from evaluation.evaluator import Evaluator

from evaluation.ensemble import (
    evaluate_ensemble,
    summarize_metric_runs,
)

from evaluation.heatmap_runner import (
    run_heatmap_mode,
    run_heatmaps_for_records,
)

from models.model_factory import ModelFactory

from training.trainer import Trainer

from shared.metrics import plot_training_history

from utils.result_manager import ResultsManager

from shared.pipelines.spectral_unsupervised import run_spectral_unsupervised

def _optional_limit(value):

    return None if value is None or value < 0 else value


def set_global_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def get_model_names(include_non_gradcam=True):

    model_names = []

    for model_name in ModelFactory.MODELS.keys():

        if not include_non_gradcam and model_name in ["vit", "vmamba"]:

            continue

        model_names.append(model_name)

    return model_names


def get_class_names(dataset_dir):

    return sorted(
        [
            folder for folder in os.listdir(dataset_dir)
            if os.path.isdir(
                os.path.join(dataset_dir, folder)
            )
        ]
    )


def run_pipeline(

    model_name,

    train_dir="data/train",

    test_dir="data/test",

    batch_size=16,

    epochs=200,

    learning_rate=1e-4,

    img_size=224,

    early_stopping_patience=20,

    early_stopping_min_delta=0.0001,

    scheduler_patience=8,

    scheduler_factor=0.5,

    scheduler_min_lr=1e-7,

    max_train_images=-1,

    max_validation_images=-1,

    max_test_images=-1,

    seed=42,

    run_index=1,
    ):

    set_global_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    # =========================
    # DATA
    # =========================

    data = DataModule(

        train_dir=train_dir,

        test_dir=test_dir,

        img_size=img_size,

        batch_size=batch_size,

        validation_split=0.2,

        max_train_images=_optional_limit(
            max_train_images
        ),

        max_validation_images=_optional_limit(
            max_validation_images
        ),

        max_test_images=_optional_limit(
            max_test_images
        ),

        seed=seed,
    )

    # =========================
    # MODEL
    # =========================

    model_wrapper = ModelFactory.create(

        model_name=model_name,

        num_classes=len(data.class_names),

        device=device,
    )

    model = model_wrapper.get_model()

    print(f"Loaded model: {model_name}")

    # =========================
    # RESULTS
    # =========================

    results = ResultsManager(
        model_wrapper.model_name
    )

    # =========================
    # TRAINER
    # =========================

    trainer = Trainer(

        model=model,

        train_loader=data.train_loader,

        validation_loader=data.validation_loader,

        device=device,

        epochs=epochs,

        learning_rate=learning_rate,

        early_stopping_patience=_optional_limit(
            early_stopping_patience
        ),

        early_stopping_min_delta=early_stopping_min_delta,

        scheduler_patience=_optional_limit(
            scheduler_patience
        ),

        scheduler_factor=scheduler_factor,

        scheduler_min_lr=scheduler_min_lr,
    )

    # =========================
    # TRAIN
    # =========================

    history = trainer.train()

    # =========================
    # PLOTS
    # =========================

    plot_training_history(
        history,
        results.plots_dir
    )

    # =========================
    # SAVE MODEL
    # =========================

    torch.save(
        model.state_dict(),
        results.get_model_path()
    )

    print("Model saved")

    # =========================
    # EVALUATION
    # =========================

    evaluator = Evaluator(

        model_wrapper=model_wrapper,

        test_loader=data.test_loader,

        class_names=data.class_names,

        device=device,

        results_manager=results,
    )

    metrics = evaluator.evaluate()

    # =========================
    # METADATA
    # =========================

    results.save_metadata(
        {

            "model": model_name,

            "epochs": epochs,

            "batch_size": batch_size,

            "learning_rate": learning_rate,

            "img_size": img_size,

            "early_stopping_patience": early_stopping_patience,

            "scheduler_patience": scheduler_patience,

            "device": str(device),

            "run_index": run_index,

            "seed": seed,
        }
    )

    print("Pipeline finished")

    return {
        "name": model_wrapper.model_name,
        "model_key": model_name,
        "run_index": run_index,
        "seed": seed,
        "results_dir": results.base_dir,
        "metrics": metrics,
        "class_names": data.class_names,
    }


def save_summary(path, model_records, ensemble_records=None):

    ensemble_records = ensemble_records or []

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    payload = {
        "models": summarize_metric_runs(
            model_records
        ),
        "ensembles": summarize_metric_runs(
            ensemble_records
        ),
        "model_runs": model_records,
        "ensemble_runs": ensemble_records,
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=4
        )

    print(f"Resumo salvo em: {path}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "heatmap", "spectral-unsupervised"]
    )

    parser.add_argument(
        "--model",
        required=False,
        type=str
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None
    )

    parser.add_argument(
        "--train-dir",
        default="data/train"
    )

    parser.add_argument(
        "--test-dir",
        default="data/test"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=8
    )

    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-7
    )

    parser.add_argument(
        "--max-train-images",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--max-validation-images",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--max-test-images",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--heatmaps-per-model",
        type=int,
        default=0
    )

    parser.add_argument(
        "--spectral-metadata-csv",
        type=str,
        default=None
    )

    parser.add_argument(
        "--max-spectral-samples",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--spectral-length",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32
    )

    parser.add_argument(
        "--ae-hidden-dim",
        type=int,
        default=256
    )

    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--ae-lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--ae-batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--ae-early-stopping-patience",
        type=int,
        default=15
    )

    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=10
    )

    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=5
    )

    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        type=str,
        default="eom",
        choices=["eom", "leaf"]
    )

    parser.add_argument(
        "--spectral-cache-dir",
        type=str,
        default="data/spectral_cache"
    )

    parser.add_argument(
        "--spectral-search-radius-arcsec",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--spectral-wavelength-min",
        type=float,
        default=3800.0
    )

    parser.add_argument(
        "--spectral-wavelength-max",
        type=float,
        default=9200.0
    )

    parser.add_argument(
        "--enable-umap",
        action="store_true"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # =========================
    # SPECTRAL MODE
    # =========================

    if args.mode == "spectral-unsupervised":

        run_spectral_unsupervised(args)
        return

    # =========================
    # HEATMAP MODE
    # =========================

    if args.model is None:

        raise ValueError(
            "--model e obrigatorio nos modos train e heatmap"
        )

    if args.mode == "heatmap":

        run_heatmap_mode(args)
        return

    # =========================
    # TRAIN MODE
    # =========================

    unique_models = get_model_names()

    if args.runs < 1:

        raise ValueError(
            "--runs deve ser maior ou igual a 1"
        )

    model_records = []

    if args.model.lower() == "all":

        for run_index in range(1, args.runs + 1):

            print("\n")
            print("=" * 70)

            print(
                f"EXECUCAO {run_index}/{args.runs}"
            )

            print("=" * 70)

            for index, model_name in enumerate(unique_models):

                try:

                    record = run_pipeline(

                        model_name=model_name,

                        train_dir=args.train_dir,

                        test_dir=args.test_dir,

                        batch_size=args.batch_size,

                        epochs=args.epochs,

                        learning_rate=args.learning_rate,

                        img_size=args.img_size,

                        early_stopping_patience=args.early_stopping_patience,

                        early_stopping_min_delta=args.early_stopping_min_delta,

                        scheduler_patience=args.scheduler_patience,

                        scheduler_factor=args.scheduler_factor,

                        scheduler_min_lr=args.scheduler_min_lr,

                        max_train_images=args.max_train_images,

                        max_validation_images=args.max_validation_images,

                        max_test_images=args.max_test_images,

                        seed=args.seed + run_index - 1,

                        run_index=run_index,
                    )

                    model_records.append(record)

                except Exception as error:

                    print(error)

                finally:

                    gc.collect()

                    if torch.cuda.is_available():

                        torch.cuda.empty_cache()

        class_names = (
            model_records[0]["class_names"]
            if model_records
            else get_class_names(args.test_dir)
        )

        ensemble_records = evaluate_ensemble(
            model_records,
            class_names
        )

        save_summary(
            os.path.join(
                "Resultados",
                "summary_all.json"
            ),
            model_records,
            ensemble_records
        )

        if args.heatmaps_per_model > 0:

            run_heatmaps_for_records(
                model_records,
                args.test_dir,
                img_size=args.img_size,
                count=args.heatmaps_per_model
            )

    else:

        for run_index in range(1, args.runs + 1):

            record = run_pipeline(

                model_name=args.model,

                train_dir=args.train_dir,

                test_dir=args.test_dir,

                batch_size=args.batch_size,

                epochs=args.epochs,

                learning_rate=args.learning_rate,

                img_size=args.img_size,

                early_stopping_patience=args.early_stopping_patience,

                early_stopping_min_delta=args.early_stopping_min_delta,

                scheduler_patience=args.scheduler_patience,

                scheduler_factor=args.scheduler_factor,

                scheduler_min_lr=args.scheduler_min_lr,

                max_train_images=args.max_train_images,

                max_validation_images=args.max_validation_images,

                max_test_images=args.max_test_images,

                seed=args.seed + run_index - 1,

                run_index=run_index,
            )

            model_records.append(record)

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        save_summary(
            os.path.join(
                "Resultados",
                f"summary_{args.model}.json"
            ),
            model_records
        )

        if args.heatmaps_per_model > 0:

            run_heatmaps_for_records(
                model_records,
                args.test_dir,
                img_size=args.img_size,
                count=args.heatmaps_per_model
            )


if __name__ == "__main__":

    main()
