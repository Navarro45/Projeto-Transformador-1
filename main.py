import argparse
import gc
import json
import os
import random

import torch

from training.data_module import DataModule

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

from hierarchy.execution_import import (
    build_hierarchy_import_plan,
    family_import_spec,
    materialize_hierarchy_import_plan,
)
from hierarchy.image_pipeline import run_hierarchical_image_pipeline

from training.trainer import Trainer

from shared.metrics import plot_training_history
from shared.stellar_hierarchy import (
    LEVEL2_CLASS_ORDER,
    OB_SPECTRAL_LABEL,
    SPECIAL_TYPE_LABEL,
    SPECIAL_STELLAR_CLASSES,
    SPECTRAL_CLASS_ORDER,
    parse_level3_class_weights,
)

from utils.result_manager import ResultsManager

from shared.pipelines.spectral_unsupervised import run_spectral_unsupervised
from shared.pipelines.star_supervised_random_forest import (
    resolve_rf_label_levels,
    run_star_random_forest,
)


def _split_manifest_for_label_level(path, label_level):

    if not path:

        return path

    root, extension = os.path.splitext(str(path))

    return f"{root}_{label_level}{extension}"


def _run_star_model_levels(args, model_names, execution_dir):

    requested_level = args.rf_label_level
    label_levels = resolve_rf_label_levels(requested_level)
    original_split_manifest = args.rf_split_manifest
    use_variants = len(label_levels) > 1
    records = []

    args.results_execution_dir = execution_dir

    try:

        for run_index in range(1, args.runs + 1):

            for model_name in model_names:

                for label_level in label_levels:

                    args.rf_label_level = label_level
                    args.rf_result_variant = (
                        label_level
                        if use_variants
                        else None
                    )
                    args.rf_split_manifest = (
                        _split_manifest_for_label_level(
                            original_split_manifest,
                            label_level,
                        )
                        if use_variants
                        else original_split_manifest
                    )

                    records.append(
                        run_star_random_forest(
                            args,
                            model_name,
                            run_index=run_index,
                        )
                    )

    finally:

        args.rf_label_level = requested_level
        args.rf_split_manifest = original_split_manifest
        args.rf_result_variant = None

    return records

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


def validate_image_splits(train_dir, validation_dir, test_dir):

    missing = []

    for split_name, split_dir in (
        ("train", train_dir),
        ("val", validation_dir),
        ("test", test_dir),
    ):

        if not os.path.isdir(split_dir):

            missing.append(f"{split_name}: diretorio ausente ({split_dir})")
            continue

        has_files = any(
            files
            for _, _, files in os.walk(split_dir)
        )

        if not has_files:

            missing.append(f"{split_name}: sem imagens ({split_dir})")

    if missing:

        details = "; ".join(missing)

        raise FileNotFoundError(
            "Os modelos de imagem exigem os splits 70/15/15 preparados. "
            f"Problemas encontrados: {details}. "
            "Execute: python dataset_scripts/prepare_dataset.py"
        )


def run_pipeline(

    model_name,

    train_dir="dataset/train",

    validation_dir="dataset/val",

    test_dir="dataset/test",

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

    execution_dir=None,
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

        validation_dir=validation_dir,

        test_dir=test_dir,

        img_size=img_size,

        batch_size=batch_size,

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
        model_wrapper.model_name,
        execution_dir=execution_dir,
        run_index=run_index,
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
        default="dataset/train"
    )

    parser.add_argument(
        "--validation-dir",
        default="dataset/val"
    )

    parser.add_argument(
        "--test-dir",
        default="dataset/test"
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
        default=256
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
        "--hierarchical",
        action="store_true",
        help=(
            "Treina a cascata em 256x256: classe do objeto, luminosidade/especial "
            "e um unico classificador espectral global no nivel 3."
        ),
    )

    parser.add_argument(
        "--hierarchy-manifest",
        default="dataset/hierarchy/roman_spectral_manifest.csv",
        help="Manifesto unico preparado por dataset_scripts/prepare_hierarchical_dataset.py.",
    )

    parser.add_argument(
        "--hierarchy-min-train-per-class",
        type=int,
        default=10,
        help="Minimo real para uma classe espectral participar do modelo global.",
    )

    parser.add_argument(
        "--hierarchy-exclude-level2-classes",
        default="",
        help=(
            "Classes removidas do nivel 2, separadas por virgula. "
            "Exemplo: --hierarchy-exclude-level2-classes III"
        ),
    )

    parser.add_argument(
        "--hierarchy-exclude-spectral-classes",
        default="",
        help=(
            "Classes espectrais removidas do nivel 3, separadas por virgula. "
            "Exemplo: --hierarchy-exclude-spectral-classes OB"
        ),
    )

    parser.add_argument(
        "--hierarchy-level3-class-weights",
        default="",
        help=(
            "Multiplicadores de peso para classes do nivel 3. "
            "Exemplo: --hierarchy-level3-class-weights OB=1.5,M=1.2"
        ),
    )

    parser.add_argument(
        "--hierarchy-balance-target",
        type=int,
        default=500,
        help="Meta maxima de amostras reais+sinteticas por saida dos niveis 2 e 3.",
    )

    parser.add_argument(
        "--hierarchy-min-synthetic-fraction",
        type=float,
        default=0.10,
        help="Fracao minima sintetica por classe; 0.10 produz 450 reais + 50 sinteticas.",
    )

    parser.add_argument(
        "--hierarchy-max-synthetic-per-source",
        type=int,
        default=10,
        help="Limite de imagens sinteticas derivadas do mesmo objeto real em cada no.",
    )

    parser.add_argument(
        "--hierarchy-disable-augmentation",
        action="store_true",
        help="Executa o baseline hierarquico somente com imagens reais.",
    )

    parser.add_argument(
        "--hierarchy-confidence-threshold",
        type=float,
        default=0.0,
        help="Confianca minima [0,1] para continuar ao proximo nivel.",
    )

    parser.add_argument(
        "--hierarchy-resume-execution",
        default=None,
        help=(
            "Diretorio Resultados/<data> existente para validar e reutilizar "
            "checkpoints hierarquicos sem retreinar nos concluidos."
        ),
    )

    parser.add_argument(
        "--hierarchy-start-level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "Nivel inicial da nova execucao hierarquica. Nos niveis 2 ou 3, "
            "copia os niveis anteriores da ultima execucao datada e treina "
            "do zero a partir do nivel escolhido."
        ),
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
        "--star-objects-index",
        type=str,
        default="dataset/star_objects/star_objects_index.csv"
    )

    parser.add_argument(
        "--rf-cache-dir",
        type=str,
        default="dataset/star_supervised/cache"
    )

    parser.add_argument(
        "--rf-split-manifest",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rf-label-level",
        type=str,
        default="main",
        choices=["main", "subclass", "both"]
    )

    parser.add_argument(
        "--rf-refresh-cache",
        action="store_true"
    )

    parser.add_argument(
        "--rf-refresh-split",
        action="store_true"
    )

    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300
    )

    parser.add_argument(
        "--rf-criterion",
        type=str,
        default="log_loss",
        choices=["gini", "entropy", "log_loss"]
    )

    parser.add_argument(
        "--rf-max-depth",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rf-min-samples-split",
        type=int,
        default=2
    )

    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=1
    )

    parser.add_argument(
        "--rf-max-features",
        type=str,
        default="sqrt"
    )

    parser.add_argument(
        "--rf-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    parser.add_argument(
        "--rf-class-weight",
        type=str,
        default="balanced_subsample"
    )

    parser.add_argument(
        "--rf-bootstrap-max-samples",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--rf-prediction-mode",
        type=str,
        default="global_probability",
        choices=["global_probability", "conditional"]
    )

    parser.add_argument(
        "--hrf-alpha",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--hrf-alpha-grid",
        type=str,
        default=""
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

    resume_execution = getattr(args, "hierarchy_resume_execution", None)
    start_level = int(getattr(args, "hierarchy_start_level", 1))
    hierarchical = bool(getattr(args, "hierarchical", False))
    raw_excluded_level2 = getattr(
        args,
        "hierarchy_exclude_level2_classes",
        "",
    )
    if isinstance(raw_excluded_level2, str):
        raw_excluded_level2 = raw_excluded_level2.split(",")
    special_aliases = {
        SPECIAL_TYPE_LABEL.upper(),
        *(label.upper() for label in SPECIAL_STELLAR_CLASSES),
    }
    excluded_level2 = set()
    for value in raw_excluded_level2 or ():
        text = str(value).strip().upper()
        if text:
            excluded_level2.add(
                SPECIAL_TYPE_LABEL if text in special_aliases else text
            )
    level2_order = LEVEL2_CLASS_ORDER
    invalid_level2 = sorted(excluded_level2.difference(level2_order))
    if invalid_level2:
        raise ValueError(
            "Classes invalidas em --hierarchy-exclude-level2-classes: "
            + ", ".join(invalid_level2)
        )
    args.hierarchy_exclude_level2_classes = tuple(
        label for label in level2_order if label in excluded_level2
    )
    raw_excluded_spectral = getattr(
        args,
        "hierarchy_exclude_spectral_classes",
        "",
    )
    if isinstance(raw_excluded_spectral, str):
        raw_excluded_spectral = raw_excluded_spectral.split(",")
    excluded_spectral = set()
    for value in raw_excluded_spectral or ():
        text = str(value).strip().upper()
        if text:
            excluded_spectral.add(
                OB_SPECTRAL_LABEL if text in {"O", "B", "OB"} else text
            )
    invalid_excluded = sorted(excluded_spectral.difference(SPECTRAL_CLASS_ORDER))
    if invalid_excluded:
        raise ValueError(
            "Classes espectrais invalidas em --hierarchy-exclude-spectral-classes: "
            + ", ".join(invalid_excluded)
        )
    args.hierarchy_exclude_spectral_classes = tuple(
        label for label in SPECTRAL_CLASS_ORDER if label in excluded_spectral
    )
    args.hierarchy_level3_class_weights = parse_level3_class_weights(
        getattr(args, "hierarchy_level3_class_weights", "")
    )
    weighted_excluded = sorted(
        set(args.hierarchy_level3_class_weights).intersection(
            args.hierarchy_exclude_spectral_classes
        )
    )
    if weighted_excluded:
        raise ValueError(
            "Uma classe nao pode ser ponderada e excluida ao mesmo tempo: "
            + ", ".join(weighted_excluded)
        )

    if resume_execution and not hierarchical:
        raise ValueError(
            "--hierarchy-resume-execution exige --hierarchical"
        )

    model_selector = args.model.lower()
    if start_level > 1 and not hierarchical:
        raise ValueError("--hierarchy-start-level 2/3 exige --hierarchical")
    if args.hierarchy_exclude_spectral_classes and not hierarchical:
        raise ValueError(
            "--hierarchy-exclude-spectral-classes exige --hierarchical"
        )
    if args.hierarchy_exclude_level2_classes and not hierarchical:
        raise ValueError("--hierarchy-exclude-level2-classes exige --hierarchical")
    if args.hierarchy_level3_class_weights and not hierarchical:
        raise ValueError("--hierarchy-level3-class-weights exige --hierarchical")
    if start_level > 1 and resume_execution:
        raise ValueError(
            "--hierarchy-start-level 2/3 nao pode ser combinado com "
            "--hierarchy-resume-execution"
        )

    hierarchy_models = None
    hierarchy_import_plan = None
    if hierarchical:

        if args.img_size != 256:
            raise ValueError("A nova hierarquia visual exige --img-size 256")

        if args.hierarchy_balance_target < 2:
            raise ValueError("--hierarchy-balance-target deve ser maior ou igual a 2")

        if not 0.0 < args.hierarchy_min_synthetic_fraction < 1.0:
            raise ValueError(
                "--hierarchy-min-synthetic-fraction deve estar entre 0 e 1"
            )

        if args.hierarchy_max_synthetic_per_source < 1:
            raise ValueError(
                "--hierarchy-max-synthetic-per-source deve ser maior ou igual a 1"
            )

        if not 0.0 <= args.hierarchy_confidence_threshold <= 1.0:
            raise ValueError("--hierarchy-confidence-threshold deve estar entre 0 e 1")

        if model_selector in {"all", "all_models"}:
            hierarchy_models = unique_models
        elif model_selector in ModelFactory.MODELS:
            hierarchy_models = [model_selector]
        else:
            raise ValueError(
                "--hierarchical aceita um modelo de imagem registrado, "
                "--model all ou --model all_models"
            )

        if start_level > 1:
            hierarchy_import_plan = build_hierarchy_import_plan(
                args=args,
                model_names=hierarchy_models,
                runs=args.runs,
            )

    if resume_execution:
        execution_dir = os.path.normpath(str(resume_execution))
        if not os.path.isdir(execution_dir):
            raise FileNotFoundError(
                f"Diretorio de retomada nao encontrado: {execution_dir}"
            )
        print(f"Retomando execucao existente: {execution_dir}")
    else:
        execution_dir = ResultsManager.create_execution_dir()

    if hierarchy_import_plan is not None:
        hierarchy_import_plan = materialize_hierarchy_import_plan(
            hierarchy_import_plan,
            execution_dir,
        )
        print(
            "Niveis anteriores importados de: "
            f"{hierarchy_import_plan['source_execution']}"
        )

    print(f"Diretorio desta execucao: {execution_dir}")

    model_records = []

    if hierarchical:

        for run_index in range(1, args.runs + 1):
            for model_name in hierarchy_models:
                model_records.append(
                    run_hierarchical_image_pipeline(
                        args=args,
                        model_name=model_name,
                        run_index=run_index,
                        execution_dir=execution_dir,
                        import_spec=family_import_spec(
                            hierarchy_import_plan,
                            model_name,
                            run_index,
                        ),
                    )
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if model_selector == "all_models":
            model_records.extend(
                _run_star_model_levels(
                    args,
                    ModelFactory.SKLEARN_MODELS,
                    execution_dir,
                )
            )

        save_summary(
            os.path.join(execution_dir, f"summary_{model_selector}_hierarchical.json"),
            model_records,
        )
        return

    if model_selector in {
        "all_star_models",
        "all_star_random_forests",
    }:

        model_records = _run_star_model_levels(
            args,
            ModelFactory.SKLEARN_MODELS,
            execution_dir,
        )

        save_summary(
            os.path.join(
                execution_dir,
                "summary_all_star_models.json"
            ),
            model_records
        )

        return

    if (
        model_selector not in {"all", "all_models"}
        and ModelFactory.is_sklearn_model(args.model)
    ):

        model_records = _run_star_model_levels(
            args,
            [args.model],
            execution_dir,
        )

        save_summary(
            os.path.join(
                execution_dir,
                f"summary_{args.model}.json"
            ),
            model_records
        )

        return

    if model_selector in {"all", "all_models"}:

        validate_image_splits(
            args.train_dir,
            args.validation_dir,
            args.test_dir,
        )

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

                        validation_dir=args.validation_dir,

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

                        execution_dir=execution_dir,
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
            class_names,
            base_output_dir=execution_dir,
        )

        all_records = list(model_records)
        summary_name = "summary_all.json"

        if model_selector == "all_models":

            star_records = _run_star_model_levels(
                args,
                ModelFactory.SKLEARN_MODELS,
                execution_dir,
            )

            all_records.extend(star_records)
            summary_name = "summary_all_models.json"

        save_summary(
            os.path.join(
                execution_dir,
                summary_name
            ),
            all_records,
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

                validation_dir=args.validation_dir,

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

                execution_dir=execution_dir,
            )

            model_records.append(record)

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        save_summary(
            os.path.join(
                execution_dir,
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
