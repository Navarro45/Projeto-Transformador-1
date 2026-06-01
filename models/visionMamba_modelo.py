import torch.nn as nn
import torch
import timm

from .modelo_base import Modelo_Base


class VisionMambaModel(Modelo_Base):

    def __init__(
        self,
        num_classes,
        device,
        variant="mambaout_tiny",
        pretrained=True
    ):

        super().__init__(
            num_classes,
            device,
            model_name="vmamba"
        )

        self.variant = variant
        self.pretrained = pretrained

        self.build_model()

    def build_model(self):

        try:
            self.model = timm.create_model(
                self.variant,
                pretrained=self.pretrained,
                num_classes=0,
            )
        except Exception as exc:
            if not self.pretrained:
                raise

            print(
                "Aviso: nao foi possivel carregar pesos pre-treinados "
                f"para {self.variant}. Usando pesos aleatorios. Erro: {exc}"
            )

            self.model = timm.create_model(
                self.variant,
                pretrained=False,
                num_classes=0,
            )

        in_features = self._infer_classifier_input_features()

        self.classifier = nn.Linear(
            in_features,
            self.num_classes
        )

        self.to(self.device)

    def forward_features(self, x):

        return self.model(x)

    def _infer_classifier_input_features(self):

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.model(dummy)

        if was_training:
            self.model.train()

        return features.shape[1]

    def get_classifier_input_features(self):

        return self.classifier.in_features

    def get_target_layer(self):

        if hasattr(self.model, "stages"):
            return self.model.stages[-1]

        if hasattr(self.model, "layers"):
            return self.model.layers[-1]

        return self.model


MambaVisionModel = VisionMambaModel
