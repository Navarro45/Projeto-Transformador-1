import torch
from PIL import Image
from torchvision import transforms
from src.model import get_model

def predict(image_path, model_path, config):
    model = get_model(config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.to(config.DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    return config.CLASS_NAMES[pred]