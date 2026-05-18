import torch
from data.data_module import DataModule
from models.model_factory import ModelFactory
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from evaluation.gradCam import GradCAMGenerator
from utils.result_manager import ResultsManager



def main():

    # ==================================================
    # CONFIG
    # ==================================================

    MODEL_NAME = "efficientnet"

    TRAIN_DIR = "data/train"

    TEST_DIR = "data/test"

    BATCH_SIZE = 16

    EPOCHS = 3

    LEARNING_RATE = 0.001

    IMG_SIZE = 224

    # TEST MODE
    MAX_TRAIN_IMAGES = 100
    MAX_VALIDATION_IMAGES = 100
    MAX_TEST_IMAGES = 100

    # ==================================================
    # DEVICE
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # ==================================================
    # DATA
    # ==================================================

    data = DataModule(

        train_dir=TRAIN_DIR,

        test_dir=TEST_DIR,

        img_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        validation_split=0.2,

        max_train_images=MAX_TRAIN_IMAGES,

        max_validation_images=MAX_VALIDATION_IMAGES,

        max_test_images=MAX_TEST_IMAGES
    )

    # ==================================================
    # MODEL
    # ==================================================

    model_wrapper = ModelFactory.create(

        model_name=MODEL_NAME,

        num_classes=len(data.class_names),

        device=device
    )

    model = model_wrapper.get_model()

    print(f"Loaded model: {MODEL_NAME}")

    # ==================================================
    # RESULTS
    # ==================================================

    results = ResultsManager(
        model_wrapper.model_name
    )

    # ==================================================
    # TRAINER
    # ==================================================

    trainer = Trainer(

        model=model,

        train_loader=data.train_loader,

        validation_loader=data.validation_loader,

        device=device,

        epochs=EPOCHS,

        learning_rate=LEARNING_RATE
    )

    trainer.train()

    # ==================================================
    # SAVE MODEL
    # ==================================================

    torch.save(

        model.state_dict(),

        f"{results.model_dir}/model.pth"
    )

    print("Model saved")

    # ==================================================
    # EVALUATION
    # ==================================================

    evaluator = Evaluator(

        model_wrapper=model_wrapper,

        test_loader=data.test_loader,

        class_names=data.class_names,

        device=device,

        results_manager=results
    )

    evaluator.evaluate()

    # ==================================================
    # SAVE METADATA
    # ==================================================

    results.save_metadata({

        "model": MODEL_NAME,

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "img_size": IMG_SIZE,

        "max_train_images": MAX_TRAIN_IMAGES,

        "max_validation_images":
            MAX_VALIDATION_IMAGES,

        "max_test_images":
            MAX_TEST_IMAGES,

        "device": str(device)
    })

    print("Metadata saved")

    # ==================================================
    # OPTIONAL GRADCAM
    # ==================================================

    """
    gradcam = GradCAMGenerator(

        model_wrapper=model_wrapper,

        transform=data.transform,

        device=device,

        results_manager=results
    )

    gradcam.generate(
        "imagem_teste.jpg"
    )
    """

    print("Pipeline finished")


if __name__ == "__main__":

    main()