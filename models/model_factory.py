from .efficientnet_modelo import EfficientNetModel
from .resnet_modelo import ResNetModel
from .convNeXt_modelo import ConvNeXtModel
from .vit_modelo import ViTModel

class ModelFactory:

    MODELS = {

        "efficientnet": EfficientNetModel,
        "resnet": ResNetModel,
        "convnext": ConvNeXtModel,
        "vit": ViTModel,
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