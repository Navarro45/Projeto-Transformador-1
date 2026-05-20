import torch
import torch.nn as nn
import torch.nn.functional as F


class Modelo_Base(nn.Module):

    def __init__(
        self,
        num_classes,
        device,
        model_name="base_model"
    ):
        super().__init__()

        self.num_classes = num_classes

        self.device = device

        self.model_name = model_name

        self.model = None

    # ==================================================
    # ABSTRACT METHODS
    # ==================================================

    def build_model(self):
        raise NotImplementedError

    def forward_features(self, x):
        raise NotImplementedError

    def get_classifier_input_features(self):
        raise NotImplementedError

    def get_target_layer(self):
        raise NotImplementedError

    # ==================================================
    # UNIVERSAL FORWARD
    # ==================================================

    def forward_with_features(self, x):

        features = self.forward_features(x)

        if len(features.shape) == 4:

            pooled = F.adaptive_avg_pool2d(
                features,
                1
            )

            embeddings = torch.flatten(
                pooled,
                1
            )

        else:

            embeddings = features

        logits = self.classifier(
            embeddings
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        return {

            "features": features,

            "embeddings": embeddings,

            "logits": logits,

            "probabilities": probabilities,

            "confidences": confidences,

            "predictions": predictions
        }

    # ==================================================
    # GET MODEL
    # ==================================================

    def get_model(self):
        return self

    # ==================================================
    # FORWARD
    # ==================================================

    def forward(self, x):

        outputs = self.forward_with_features(x)

        return outputs["logits"]

    def to(self, device):

        self.device = device

        return super().to(device)
