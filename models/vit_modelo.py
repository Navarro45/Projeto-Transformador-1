import torch.nn as nn
from torchvision import models
from .modelo_base import Modelo_Base


class ViTModel(Modelo_Base):

    def __init__(
        self,
        num_classes,
        device,
        image_size=256,
    ):

        super().__init__(
            num_classes,
            device,
            model_name="vit_b_16",
            image_size=image_size,
        )

        self.build_model()

    def build_model(self):

        weights = models.ViT_B_16_Weights.DEFAULT
        if self.image_size == 224:
            self.model = models.vit_b_16(weights=weights)
        else:
            self.model = models.vit_b_16(
                weights=None,
                image_size=self.image_size,
            )
            state = weights.get_state_dict(progress=True)
            state = models.vision_transformer.interpolate_embeddings(
                image_size=self.image_size,
                patch_size=16,
                model_state=state,
                reset_heads=True,
            )
            self.model.load_state_dict(state, strict=False)

        in_features = (
            self.model.heads.head.in_features
        )

        self.classifier = nn.Linear(
            in_features,
            self.num_classes
        )

        self.model.heads = nn.Identity()

        self.model = self.model.to(
            self.device
        )

        self.classifier = self.classifier.to(
            self.device
        )

    def forward_features(self, x):

        return self.model(x)

    def get_target_layer(self):

        return self.model.encoder.layers[-1]
