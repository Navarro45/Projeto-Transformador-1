import torch.nn as nn
from mambavision import create_model
from .modelo_base import BaseVisionModel


class MambaVisionModel(BaseVisionModel):

    def __init__(
        self,
        num_classes,
        device,
        variant="mamba_vision_T"
    ):

        super().__init__(
            num_classes,
            device,
            model_name=variant
        )

        self.variant = variant

        self.build_model()

    def build_model(self):

        # num_classes=0 remove classifier original
        self.model = create_model(

            self.variant,

            pretrained=True
        )

        # Descobrir tamanho automaticamente
        dummy_features = self.model.forward_features

        # Para Tiny normalmente 640
        in_features = 640

        self.classifier = nn.Linear(
            in_features,
            self.num_classes
        )

        self.model.head = nn.Identity()

        self.model = self.model.to(
            self.device
        )

        self.classifier = self.classifier.to(
            self.device
        )

    def forward_features(self, x):

        return self.model(x)

    def get_target_layer(self):

        return self.model.levels[-1]