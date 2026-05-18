import torch.nn as nn
from modelo_base import Modelo_Base
from mambavision import MambaVision


class MambaVisionModel(Modelo_Base):

    def __init__(
        self,
        num_classes,
        device
    ):

        super().__init__(
            num_classes,
            device,
            model_name="mamba_vision"
        )

        self.build_model()

    def build_model(self):

        self.model = MambaVision(
            num_classes=0
        )

        in_features = 768

        self.classifier = nn.Linear(
            in_features,
            self.num_classes
        )

        self.model = self.model.to(
            self.device
        )

        self.classifier = self.classifier.to(
            self.device
        )

    def forward_features(self, x):

        return self.model(x)

    def get_target_layer(self):

        return self.model.layers[-1]