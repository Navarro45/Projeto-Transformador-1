import torch
import torch.nn as nn

class HierarchicalModel(nn.Module):
    def __init__(self, image_model, spectral_models):
        super().__init__()
        self.image_model = image_model
        self.spectral_models = spectral_models

    def forward(self, image, spectrum=None):
        main_logits = self.image_model(image)
        main_pred = torch.argmax(main_logits, dim=1)

        subtype_logits = None

        if spectrum is not None:
            subtype_logits = []

            for i in range(len(main_pred)):
                cls = main_pred[i].item()

                if spectrum[i] is None:
                    subtype_logits.append(None)
                    continue

                model = self.spectral_models[cls]
                spec = spectrum[i].unsqueeze(0)
                out = model(spec)
                subtype_logits.append(out)

        return main_logits, subtype_logits