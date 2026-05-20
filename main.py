import argparse

import torch

from data.data_module import DataModule
from evaluation.evaluator import Evaluator
from models.model_factory import ModelFactory
from training.trainer import Trainer
from utils.result_manager import ResultsManager


def _optional_limit(value):
    return None if value is None or value < 0 else value


def run_pipeline(
    model_name="vmamba",
    train_dir="data/train",
    test_dir="data/test",
    batch_size=16,
    epochs=3,
    learning_rate=0.001,
    img_size=224,
    max_train_images=100,
    max_validation_images=100,
    max_test_images=100,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    data = DataModule(
        train_dir=train_dir,
        test_dir=test_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        max_train_images=_optional_limit(max_train_images),
        max_validation_images=_optional_limit(max_validation_images),
        max_test_images=_optional_limit(max_test_images),
    )

    model_wrapper = ModelFactory.create(
        model_name=model_name,
        num_classes=len(data.class_names),
        device=device,
    )

    model = model_wrapper.get_model()

    print(f"Loaded model: {model_name}")

    results = ResultsManager(
        model_wrapper.model_name
    )

    trainer = Trainer(
        model=model,
        train_loader=data.train_loader,
        validation_loader=data.validation_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    trainer.train()

    torch.save(
        model.state_dict(),
        f"{results.model_dir}/model.pth",
    )

    print("Model saved")

    evaluator = Evaluator(
        model_wrapper=model_wrapper,
        test_loader=data.test_loader,
        class_names=data.class_names,
        device=device,
        results_manager=results,
    )

    evaluator.evaluate()

    results.save_metadata(
        {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "img_size": img_size,
            "max_train_images": _optional_limit(max_train_images),
            "max_validation_images": _optional_limit(max_validation_images),
            "max_test_images": _optional_limit(max_test_images),
            "device": str(device),
        }
    )

    print("Metadata saved")
    print("Pipeline finished")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina e avalia um dos modelos disponiveis."
    )
    parser.add_argument("--model", default="vmamba")
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-train-images", type=int, default=100)
    parser.add_argument("--max-validation-images", type=int, default=100)
    parser.add_argument("--max-test-images", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        model_name=args.model,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        max_train_images=args.max_train_images,
        max_validation_images=args.max_validation_images,
        max_test_images=args.max_test_images,
    )


if __name__ == "__main__":
    main()
