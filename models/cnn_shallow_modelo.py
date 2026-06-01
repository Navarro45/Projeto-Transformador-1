import torch.nn as nn

from .modelo_base import Modelo_Base


class CNNShallowModel(Modelo_Base):

    def __init__(
        self,
        num_classes,
        device
    ):

        super().__init__(
            num_classes,
            device,
            model_name="cnn_shallow"
        )

        self.build_model()

    def build_model(self):

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, self.num_classes)
        )

        self.to(self.device)

    def forward_features(self, x):

        return self.features(x)

    def get_classifier_input_features(self):

        return 128

    def get_target_layer(self):

        return self.features[8]

    def supports_gradcam(self):

        return True
