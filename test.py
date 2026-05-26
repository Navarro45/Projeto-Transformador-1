import torch

from PIL import Image

from torchvision import transforms

from utils.modelo_loader import (
    ModelLoader
)

# ==================================================
# DEVICE
# ==================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

# ==================================================
# LOAD MODEL
# ==================================================

model_wrapper = ModelLoader.load_model(

    results_folder=(
        "Resultados/vit_b_16/2026-05-26_19-09-10"
    ),

    device=device
)

# ==================================================
# TRANSFORM
# ==================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])

# ==================================================
# LOAD IMAGE
# ==================================================

image = Image.open(
    "data/test/galaxy/galaxy_7.jpg"
).convert("RGB")

tensor = transform(image)

tensor = tensor.unsqueeze(0).to(device)

# ==================================================
# PREDICTION
# ==================================================

outputs = (
    model_wrapper
    .forward_with_features(tensor)
)

prediction = (
    outputs["predictions"]
    .item()
)

confidence = (
    outputs["confidences"]
    .item()
)

print(f"Prediction: {prediction}")

print(f"Confidence: {confidence:.4f}")