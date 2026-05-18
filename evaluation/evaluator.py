import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix,classification_report,ConfusionMatrixDisplay
)


class Evaluator:

    def __init__(
        self,
        model_wrapper,
        test_loader,
        class_names,
        device,
        results_manager
    ):

        self.model_wrapper = model_wrapper

        self.model = model_wrapper.get_model()

        self.test_loader = test_loader

        self.class_names = class_names

        self.device = device

        self.results = results_manager

    def evaluate(self):

        self.model.eval()

        y_true = []
        y_pred = []

        logits_all = []
        probs_all = []
        confs_all = []
        embeddings_all = []

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(self.device)

                outputs = (
                    self.model_wrapper
                    .forward_with_features(images)
                )

                y_true.extend(labels.numpy())

                y_pred.extend(
                    outputs["predictions"]
                    .cpu()
                    .numpy()
                )

                logits_all.append(
                    outputs["logits"]
                    .cpu()
                    .numpy()
                )

                probs_all.append(
                    outputs["probabilities"]
                    .cpu()
                    .numpy()
                )

                confs_all.append(
                    outputs["confidences"]
                    .cpu()
                    .numpy()
                )

                embeddings_all.append(
                    outputs["embeddings"]
                    .cpu()
                    .numpy()
                )

        logits_all = np.concatenate(logits_all)

        probs_all = np.concatenate(probs_all)

        confs_all = np.concatenate(confs_all)

        embeddings_all = np.concatenate(
            embeddings_all
        )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names
        )

        # =========================
        # SAVE ARRAYS
        # =========================

        np.save(
            os.path.join(
                self.results.predictions_dir,
                "logits.npy"
            ),
            logits_all
        )

        np.save(
            os.path.join(
                self.results.predictions_dir,
                "probabilities.npy"
            ),
            probs_all
        )

        np.save(
            os.path.join(
                self.results.predictions_dir,
                "confidences.npy"
            ),
            confs_all
        )

        np.save(
            os.path.join(
                self.results.predictions_dir,
                "predictions.npy"
            ),
            y_pred
        )

        np.save(
            os.path.join(
                self.results.predictions_dir,
                "labels.npy"
            ),
            y_true
        )

        np.save(
            os.path.join(
                self.results.predictions_dir,
                "embeddings.npy"
            ),
            embeddings_all
        )

        # =========================
        # CONFUSION MATRIX
        # =========================

        np.savetxt(
            os.path.join(
                self.results.metrics_dir,
                "confusion_matrix.csv"
            ),
            cm,
            delimiter=",",
            fmt="%d"
        )

        plt.figure(figsize=(10, 10))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )

        disp.plot(cmap="Blues")

        plt.savefig(
            os.path.join(
                self.results.metrics_dir,
                "confusion_matrix.png"
            )
        )

        plt.close()

        # =========================
        # REPORT
        # =========================

        with open(
            os.path.join(
                self.results.metrics_dir,
                "classification_report.txt"
            ),
            "w"
        ) as f:

            f.write(report)

        print("Evaluation saved")