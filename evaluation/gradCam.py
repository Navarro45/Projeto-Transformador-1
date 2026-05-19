import os
import cv2
import numpy as np

from PIL import Image

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import (show_cam_on_image)

from pytorch_grad_cam.utils.model_targets import (ClassifierOutputTarget)


class GradCAMGenerator:

    def __init__(
        self,
        model_wrapper,
        transform,
        device,
        results_manager,
        img_size=224
    ):

        self.model_wrapper = model_wrapper

        self.model = model_wrapper.get_model()

        self.transform = transform

        self.device = device

        self.results = results_manager

        self.img_size = img_size

    def generate(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = self.transform(image)

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

        rgb_img = np.array(
            image.resize(
                (self.img_size, self.img_size)
            )
        ) / 255.0

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