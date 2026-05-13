import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class EfficientNetClassifier:

    def __init__(
        self,
        train_dir,
        test_dir,
        validation_split=0.2,
        output_dir="output",
        img_size=224,
        batch_size=32,
        epochs=5,
        learning_rate=0.001
    ):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.validation_split = validation_split
        self.output_dir = output_dir

        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        self._load_data()
        self._build_model()

    # ==================================================
    # DATA
    # ==================================================

    def _load_data(self):

        full_train_dataset = datasets.ImageFolder(
            self.train_dir,
            transform=self.transform
        )

        self.test_dataset = datasets.ImageFolder(
            self.test_dir,
            transform=self.transform
        )

        train_size = int(
            (1 - self.validation_split) * len(full_train_dataset)
        )

        validation_size = len(full_train_dataset) - train_size

        self.train_dataset, self.validation_dataset = random_split(
            full_train_dataset,
            [train_size, validation_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.class_names = full_train_dataset.classes
        self.num_classes = len(self.class_names)

    # ==================================================
    # MODEL
    # ==================================================

    def _build_model(self):

        self.model = models.efficientnet_b0(weights="DEFAULT")

        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            self.num_classes
        )

        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

    # ==================================================
    # TRAIN
    # ==================================================

    def train(self):

        self.train_losses = []
        self.train_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

        for epoch in range(self.epochs):

            self.model.train()

            running_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=True
            )

            for images, labels in progress_bar:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{(correct/total):.4f}"
                })

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            validation_loss, validation_accuracy = self.validate()

            self.validation_losses.append(validation_loss)
            self.validation_accuracies.append(validation_accuracy)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {epoch_loss:.4f}")
            print(f"Train Accuracy: {epoch_acc:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # ==================================================
    # VALIDATION
    # ==================================================

    def validate(self):

        self.model.eval()

        validation_loss = 0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for val_images, val_labels in self.validation_loader:

                val_images = val_images.to(self.device)
                val_labels = val_labels.to(self.device)

                val_outputs = self.model(val_images)

                val_loss = self.criterion(
                    val_outputs,
                    val_labels
                )

                validation_loss += val_loss.item()

                _, val_predicted = torch.max(val_outputs, 1)

                validation_total += val_labels.size(0)

                validation_correct += (
                    val_predicted == val_labels
                ).sum().item()

        validation_loss /= len(self.validation_loader)
        validation_accuracy = validation_correct / validation_total

        return validation_loss, validation_accuracy
    
    # ==================================================
    # EVALUATE
    # ==================================================

    def evaluate(self):

        self.model.eval()

        self.y_true = []
        self.y_pred = []

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                self.y_true.extend(labels.numpy())
                self.y_pred.extend(predicted.cpu().numpy())

        print("Evaluation complete")

    # ==================================================
    # SAVE METRICS
    # ==================================================

    def save_metrics(self):

        cm = confusion_matrix(self.y_true, self.y_pred)

        np.savetxt(
            os.path.join(self.output_dir, "confusion_matrix.csv"),
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

        plt.title("Matriz de Confusão")

        plt.savefig(
            os.path.join(self.output_dir, "confusion_matrix.png")
        )

        plt.close()

        report = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.class_names
        )

        with open(
            os.path.join(self.output_dir, "classification_report.txt"),
            "w"
        ) as f:
            f.write(report)

        plt.figure(figsize=(8, 5))

        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.validation_accuracies, label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy durante treinamento")

        plt.savefig(
            os.path.join(self.output_dir, "accuracy_curve.png")
        )

        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.validation_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss durante treinamento")

        plt.savefig(
            os.path.join(self.output_dir, "loss_curve.png")
        )

        plt.close()

        print("Metrics saved")

    # ==================================================
    # SAVE TRAINING DATA
    # ==================================================

    def save_training_data(self):

        train_data = {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "class_names": self.class_names
        }

        torch.save(
            train_data,
            os.path.join(
                self.output_dir,
                "training_data.pth"
            )
        )

        print("Training data saved")

    # ==================================================
    # LOAD TRAINING DATA
    # ==================================================

    def load_training_data(self):

        data = torch.load(
            os.path.join(
                self.output_dir,
                "training_data.pth"
            ),
            map_location=self.device
        )

        self.train_losses = data["train_losses"]
        self.train_accuracies = data["train_accuracies"]
        self.y_true = data["y_true"]
        self.y_pred = data["y_pred"]
        self.class_names = data["class_names"]

        print("Training data loaded")

    # ==================================================
    # TEST EVALUATION SUMMARY
    # ==================================================

    def evaluate_test_results(self):

        report_dict = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        output_file = os.path.join(
            self.output_dir,
            "test_evaluation_summary.txt"
        )
        with open(output_file, "w") as f:

            f.write("=== TEST DATA EVALUATION ===\n")

            accuracy = report_dict["accuracy"]
            macro_avg = report_dict["macro avg"]
            weighted_avg = report_dict["weighted avg"]

            f.write(f"Overall Accuracy: {accuracy:.4f}\n")

            f.write("=== MACRO AVERAGE ===\n")
            f.write(f"Precision: {macro_avg['precision']:.4f}\n")
            f.write(f"Recall: {macro_avg['recall']:.4f}\n")
            f.write(f"F1-Score: {macro_avg['f1-score']:.4f}\n")
            f.write("=== WEIGHTED AVERAGE ===\n")
            f.write(f"Precision: {weighted_avg['precision']:.4f}\n")
            f.write(f"Recall: {weighted_avg['recall']:.4f}\n")
            f.write(f"F1-Score: {weighted_avg['f1-score']:.4f}\n")
            f.write("=== PER CLASS RESULTS ===\n")

            for class_name in self.class_names:
                metrics = report_dict[class_name]
                f.write(f"Class: {class_name}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
        print(f"Test evaluation saved in: {output_file}")
    # ==================================================
    # SAVE MODEL
    # ==================================================

    def save_model(self):

        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.output_dir,
                "efficientnet_model.pth"
            )
        )

        print("Model saved")

    # ==================================================
    # LOAD MODEL
    # ==================================================

    def load_model(self, model_path):

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.eval()

        print("Model loaded")

    # ==================================================
    # PREDICT SINGLE IMAGE
    # ==================================================

    def predict_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():

            output = self.model(tensor)

            predicted = output.argmax(dim=1).item()

        predicted_class = self.class_names[predicted]

        print(f"Prediction: {predicted_class}")

        return predicted_class
    

    # ==================================================
    # GRADCAM
    # ==================================================

    def generate_gradcam(self, image_path):

        image = Image.open(image_path).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        target_layers = [self.model.features[-1]]

        cam = GradCAM(
            model=self.model,
            target_layers=target_layers
        )

        self.model.eval()

        with torch.no_grad():

            output = self.model(tensor)
            predicted_class = output.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(predicted_class)]

        grayscale_cam = cam(
            input_tensor=tensor,
            targets=targets
        )

        grayscale_cam = grayscale_cam[0]

        rgb_img = np.array(
            image.resize((self.img_size, self.img_size))
        ) / 255.0

        visualization = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )

        output_path = os.path.join(
            self.output_dir,
            "gradcam_result.png"
        )

        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        print(f"GradCAM salvo em: {output_path}")