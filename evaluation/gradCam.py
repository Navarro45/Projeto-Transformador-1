import os
import cv2
import numpy as np

from PIL import Image

from utils.crop_and_pad import (
    CENTER_CROP_RATIO,
    crop_and_pad_image,
)


class GradCAMGenerator:

    def __init__(
        self,
        model_wrapper,
        transform,
        device,
        results_manager,
        img_size=224,
        use_crop_and_pad=True,
        center_crop_ratio=CENTER_CROP_RATIO
    ):

        self.model_wrapper = model_wrapper

        self.model = model_wrapper.get_model()

        self.transform = transform

        self.device = device

        self.results = results_manager

        self.img_size = img_size

        self.use_crop_and_pad = use_crop_and_pad

        self.center_crop_ratio = center_crop_ratio

    def _prepare_image(self, image):

        if not self.use_crop_and_pad:

            return image.resize(
                (self.img_size, self.img_size)
            )

        return crop_and_pad_image(
            image,
            image_size=self.img_size,
            center_crop_ratio=self.center_crop_ratio,
        )

    def _load_gradcam_dependencies(self):

        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Grad-CAM requer o pacote `grad-cam`. "
                "Instale com `pip install grad-cam`."
            ) from exc

        return (
            GradCAM,
            show_cam_on_image,
            ClassifierOutputTarget
        )

    def generate(self, image_path):

        (
            GradCAM,
            show_cam_on_image,
            ClassifierOutputTarget
        ) = self._load_gradcam_dependencies()

        image = Image.open(
            image_path
        ).convert("RGB")

        heatmap_image = self._prepare_image(
            image
        )

        tensor = self.transform(
            heatmap_image
        )

        tensor = tensor.unsqueeze(0).to(
            self.device
        )

        target_layers = [
            self.model_wrapper.get_target_layer()
        ]

        cam = GradCAM(
            model=self.model,
            target_layers=target_layers
        )

        outputs = (
            self.model_wrapper
            .forward_with_features(tensor)
        )

        predicted_class = (
            outputs["predictions"]
            .item()
        )

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=tensor,
            targets=targets
        )[0]

        rgb_img = (
            np.array(heatmap_image) /
            255.0
        )

        visualization = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )

        image_name = os.path.basename(
            image_path
        )

        output_path = os.path.join(
            self.results.gradcam_dir,
            f"gradcam_{image_name}"
        )

        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        print(f"GradCAM salvo: {output_path}")
