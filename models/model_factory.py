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
        device,
        image_size=256,
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
                device,
                image_size=image_size,
            )
        )

    @staticmethod
    def create_for_node(
        model_name,
        node_key,
        num_classes,
        device,
        image_size=256,
    ):
        """Cria uma instancia independente da familia para um nivel hierarquico."""
        model = ModelFactory.create(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            image_size=image_size,
        )
        model.hierarchy_node = str(node_key)
        return model

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
