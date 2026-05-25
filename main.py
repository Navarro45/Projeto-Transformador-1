import argparse
import gc

import torch

from data.data_module import DataModule

from evaluation.evaluator import Evaluator

from models.model_factory import ModelFactory

from training.trainer import Trainer

from shared.metrics import plot_training_history

from utils.result_manager import ResultsManager


def _optional_limit(value):

    return None if value is None or value < 0 else value


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
):

    if not model_name:

        raise ValueError(
            "O nome do modelo deve ser informado."
        )

    # =========================
    # DEVICE
    # =========================

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

    results.summary()

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
    # TRAINING PLOTS
    # =========================

    plot_training_history(
        history,
        results.plots_dir
    )

    print("Training plots saved")

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

    evaluator.evaluate()

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

            "early_stopping_patience": _optional_limit(
                early_stopping_patience
            ),

            "early_stopping_min_delta": early_stopping_min_delta,

            "scheduler": "ReduceLROnPlateau",

            "scheduler_patience": _optional_limit(
                scheduler_patience
            ),

            "scheduler_factor": scheduler_factor,

            "scheduler_min_lr": scheduler_min_lr,

            "max_train_images": _optional_limit(
                max_train_images
            ),

            "max_validation_images": _optional_limit(
                max_validation_images
            ),

            "max_test_images": _optional_limit(
                max_test_images
            ),

            "device": str(device),
        }
    )

    print("Metadata saved")

    print("Pipeline finished")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Treina e avalia modelos."
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Nome do modelo ou 'all'."
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

    return parser.parse_args()


def main():

    args = parse_args()

    # =========================
    # REMOVE ALIASES DUPLICADOS
    # =========================

    unique_models = []

    for model_name in ModelFactory.MODELS.keys():

        normalized = model_name.lower()

        if normalized == "visionmamba":
            continue

        unique_models.append(model_name)

    # =========================
    # EXECUTA TODOS
    # =========================

    if args.model.lower() == "all":

        total_models = len(unique_models)

        # TESTES QUE VOCÊ QUER REALIZAR
        scheduler_tests = [3,5,8]

        for scheduler_patience in scheduler_tests:

            print("\n")
            print("=" * 70)

            print(
                f"TESTANDO scheduler_patience="
                f"{scheduler_patience}"
            )

            print("=" * 70)

            for index, model_name in enumerate(unique_models):

                print("\n")
                print("=" * 70)

                print(
                    f"[{index + 1}/{total_models}] "
                    f"Executando modelo: {model_name}"
                )

                print("=" * 70)

                try:

                    run_pipeline(

                        model_name=model_name,

                        train_dir=args.train_dir,

                        test_dir=args.test_dir,

                        batch_size=args.batch_size,

                        epochs=args.epochs,

                        learning_rate=args.learning_rate,

                        img_size=args.img_size,

                        early_stopping_patience=args.early_stopping_patience,

                        early_stopping_min_delta=args.early_stopping_min_delta,

                        # AQUI ALTERA AUTOMATICAMENTE
                        scheduler_patience=scheduler_patience,

                        scheduler_factor=args.scheduler_factor,

                        scheduler_min_lr=args.scheduler_min_lr,

                        max_train_images=args.max_train_images,

                        max_validation_images=args.max_validation_images,

                        max_test_images=args.max_test_images,
                    )

                    print(
                        f"\nModelo {model_name} "
                        f"finalizado com sucesso"
                    )

                except Exception as error:

                    print("\n")
                    print("!" * 70)

                    print(
                        f"ERRO AO EXECUTAR O MODELO: "
                        f"{model_name}"
                    )

                    print(error)

                    print("!" * 70)

                finally:

                    gc.collect()

                    if torch.cuda.is_available():

                        torch.cuda.empty_cache()

        print("\n")
        print("=" * 70)
        print("TODOS OS MODELOS FORAM PROCESSADOS")
        print("=" * 70)

    # =========================
    # EXECUTA UM ÚNICO MODELO
    # =========================

    else:

        run_pipeline(

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
        )


if __name__ == "__main__":

    main()