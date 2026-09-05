from .efficientnet_modelo import EfficientNetModel
from .resnet_modelo import ResNetModel
from .convNeXt_modelo import ConvNeXtModel
from .vit_modelo import ViTModel
from .visionMamba_modelo import VisionMambaModel
from .cnn_shallow_modelo import CNNShallowModel
from .random_forest_modelo import (
    FlatRandomForestModel,
    HierarchicalRandomForestModel,
)

class ModelFactory:

    MODELS = {

        "efficientnet": EfficientNetModel,
        "resnet": ResNetModel,
        "convnext": ConvNeXtModel,
        "vit": ViTModel,
        "vmamba": VisionMambaModel,
        "cnn_shallow": CNNShallowModel,
    }

    SKLEARN_MODELS = {

        "flat_random_forest": FlatRandomForestModel,
        "hierarchical_random_forest": HierarchicalRandomForestModel,
    }

    @staticmethod
    def create(
        model_name,
        num_classes,
        device
    ):

        if model_name not in (
            ModelFactory.MODELS
        ):

            raise ValueError(
                f"Modelo {model_name} não suportado"
            )

        return (
            ModelFactory.MODELS[model_name](
                num_classes,
                device
            )
        )

    @staticmethod
    def is_sklearn_model(model_name):

        return model_name in ModelFactory.SKLEARN_MODELS

    @staticmethod
    def create_sklearn(
        model_name,
        config=None
    ):

        if model_name not in (
            ModelFactory.SKLEARN_MODELS
        ):

            raise ValueError(
                f"Modelo sklearn {model_name} nao suportado"
            )

        return (
            ModelFactory.SKLEARN_MODELS[model_name](
                config
            )
        )
