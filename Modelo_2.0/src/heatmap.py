import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        selected = logits[torch.arange(logits.size(0)), class_idx]
        self.model.zero_grad()
        selected.sum().backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def _to_numpy_image(tensor_img):
    img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0.0, 1.0)
    return img


def _upsample_cam_to_image(cam_2d, height, width):
    """Alinha espacialmente o CAM à imagem (ex.: 7×7 → 224×224)."""
    t = torch.as_tensor(cam_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(height, width), mode="bicubic", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


def save_gradcam_samples(model, dataloader, config, class_names):
    os.makedirs(config.PATHS["heatmaps"], exist_ok=True)
    device = config.DEVICE
    max_images = getattr(config, "HEATMAP_MAX_IMAGES", 24)

    target_layer = model.model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    model.eval()
    saved = 0

    with torch.enable_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            cams = gradcam(images)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for i in range(images.size(0)):
                if saved >= max_images:
                    return

                img = _to_numpy_image(images[i])
                h, w = img.shape[0], img.shape[1]
                cam_small = cams[i, 0].detach().cpu().numpy()
                cam = _upsample_cam_to_image(cam_small, h, w)

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img, aspect="equal", interpolation="bilinear")
                ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0, interpolation="bicubic")
                ax.axis("off")

                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]
                ax.set_title(f"true={true_label} | pred={pred_label}")

                out_path = os.path.join(config.PATHS["heatmaps"], f"gradcam_{saved:03d}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                saved += 1
