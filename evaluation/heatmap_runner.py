import gc
import glob
import os
import random
from pathlib import Path

import torch
from torchvision import transforms

from evaluation.gradCam import GradCAMGenerator
from models.model_factory import ModelFactory
from utils.result_manager import ResultsManager


RESULTS_MAPPING = {
    "efficientnet": "efficientnet_b0",
    "resnet": "resnet50",
    "convnext": "convnext_tiny",
    "vit": "vit_b_16",
    "vmamba": "vmamba",
    "cnn_shallow": "cnn_shallow",
}


def count_classes(dataset_dir):

    return len(
        [
            folder for folder in os.listdir(dataset_dir)
            if os.path.isdir(
                os.path.join(dataset_dir, folder)
            )
        ]
    )


def get_random_image(dataset_dir):

    return get_random_images(
        dataset_dir,
        count=1
    )[0]


def get_random_images(dataset_dir, count):

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp"
    ]

    image_paths = []

    for ext in extensions:

        image_paths.extend(
            glob.glob(
                os.path.join(
                    dataset_dir,
                    "**",
                    ext
                ),
                recursive=True
            )
        )

    if not image_paths:

        raise ValueError(
            f"Nenhuma imagem encontrada em "
            f"{dataset_dir}"
        )

    if count <= len(image_paths):

        return random.sample(
            image_paths,
            count
        )

    return random.choices(
        image_paths,
        k=count
    )


def get_latest_weights(model_name):

    if model_name not in RESULTS_MAPPING:

        raise ValueError(
            f"Modelo desconhecido: {model_name}"
        )

    model_dir = (
        Path("Resultados") /
        RESULTS_MAPPING[model_name]
    )

    if not model_dir.exists():

        raise ValueError(
            f"Nenhum resultado encontrado "
            f"para {model_name}"
        )

    checkpoints = list(
        model_dir.glob(
            "**/model/model.pth"
        )
    )

    if not checkpoints:

        raise ValueError(
            f"Nenhum checkpoint encontrado "
            f"para {model_name}"
        )

    return str(
        max(
            checkpoints,
            key=os.path.getmtime
        )
    )


def model_supports_gradcam(model_name, test_dir, device):

    if model_name in ["vit", "vmamba"]:

        return False

    model_wrapper = ModelFactory.create(
        model_name=model_name,
        num_classes=count_classes(test_dir),
        device=device,
    )

    return model_wrapper.supports_gradcam()


def run_heatmap(
    model_name,
    weights_path,
    test_dir,
    img_size=224,
    image_path=None,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if image_path is None:

        image_path = get_random_image(
            test_dir
        )

    print(f"Random image selected: {image_path}")

    num_classes = count_classes(
        test_dir
    )

    print(f"Detected classes: {num_classes}")

    if model_name in ["vit", "vmamba"]:

        raise ValueError(
            f"Modelo {model_name} nao suporta Grad-CAM "
            "nesta etapa."
        )

    model_wrapper = ModelFactory.create(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
    )

    model = model_wrapper.get_model()

    if not model_wrapper.supports_gradcam():

        raise ValueError(
            f"Modelo {model_name} nao suporta Grad-CAM "
            "nesta etapa."
        )

    print(f"Loading weights: {weights_path}")

    model.load_state_dict(
        torch.load(
            weights_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    checkpoint_dir = Path(
        weights_path
    ).parent.parent

    results = ResultsManager.from_existing(
        checkpoint_dir
    )

    transform = transforms.Compose([
        transforms.Resize(
            (img_size, img_size)
        ),
        transforms.ToTensor(),
    ])

    gradcam = GradCAMGenerator(
        model_wrapper=model_wrapper,
        transform=transform,
        device=device,
        results_manager=results,
        img_size=img_size,
    )

    gradcam.generate(image_path)

    print("Heatmap generated")


def run_heatmaps(
    model_name,
    weights_path,
    test_dir,
    img_size=224,
    count=1,
):

    for image_index, image_path in enumerate(
        get_random_images(
            test_dir,
            count
        ),
        start=1
    ):

        print(
            f"Heatmap {image_index}/{count}: "
            f"{image_path}"
        )

        run_heatmap(
            model_name=model_name,
            weights_path=weights_path,
            test_dir=test_dir,
            img_size=img_size,
            image_path=image_path,
        )


def run_heatmaps_for_records(
    model_records,
    test_dir,
    img_size=224,
    count=1,
):

    latest_by_model = {}

    for record in model_records:

        latest_by_model[record["model_key"]] = record

    for model_name, record in latest_by_model.items():

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if not model_supports_gradcam(
            model_name,
            test_dir,
            device
        ):

            print(
                f"Pulando heatmaps de {model_name}: "
                "Grad-CAM nao suportado."
            )

            continue

        weights_path = os.path.join(
            record["results_dir"],
            "model",
            "model.pth"
        )

        run_heatmaps(
            model_name=model_name,
            weights_path=weights_path,
            test_dir=test_dir,
            img_size=img_size,
            count=count,
        )


def run_heatmap_mode(args):

    heatmap_count = max(
        1,
        args.heatmaps_per_model
    )

    if args.model.lower() == "all":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model_names = list(
            ModelFactory.MODELS.keys()
        )

        total_models = len(model_names)

        print("\n")
        print("=" * 70)
        print(
            f"GERANDO HEATMAPS "
            f"DE {total_models} MODELOS"
        )
        print("=" * 70)

        for index, model_name in enumerate(model_names):

            print("\n")
            print("=" * 70)
            print(
                f"[{index + 1}/{total_models}] "
                f"Gerando heatmap: {model_name}"
            )
            print("=" * 70)

            try:

                if not model_supports_gradcam(
                    model_name,
                    args.test_dir,
                    device
                ):

                    print(
                        f"Pulando {model_name}: "
                        "Grad-CAM nao suportado."
                    )

                    continue

                weights_path = get_latest_weights(
                    model_name
                )

                print(
                    f"Checkpoint encontrado: "
                    f"{weights_path}"
                )

                run_heatmaps(
                    model_name=model_name,
                    weights_path=weights_path,
                    test_dir=args.test_dir,
                    img_size=args.img_size,
                    count=heatmap_count,
                )

                print(
                    f"Heatmap gerado "
                    f"para {model_name}"
                )

            except Exception as error:

                print("\n")
                print("!" * 70)
                print(
                    f"ERRO AO GERAR HEATMAP: "
                    f"{model_name}"
                )
                print(error)
                print("!" * 70)

            finally:

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

        return

    if args.weights is None:

        raise ValueError(
            "--weights e obrigatorio "
            "quando usar um unico modelo"
        )

    run_heatmaps(
        model_name=args.model,
        weights_path=args.weights,
        test_dir=args.test_dir,
        img_size=args.img_size,
        count=heatmap_count,
    )
