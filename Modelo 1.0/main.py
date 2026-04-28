from src.dataset import get_dataloaders
from src.model import get_model, unfreeze_model
from src.train import train
from src.evaluate import test
from src.utils import set_seed, ensure_dirs
import src.config as config


##Este modelo está entrando em overfitting e ainda não foi corrigido



def main():
    set_seed(config.SEED)

    ensure_dirs([
        config.MODEL_DIR,
        config.LOG_DIR,
        config.METRIC_DIR,
        config.PLOT_DIR
    ])

    train_loader, val_loader, test_loader = get_dataloaders(
        config.DATA_DIR,
        config.BATCH_SIZE
    )

    model = get_model(config.NUM_CLASSES)

    print("Treinando camada final...")
    train(model, train_loader, val_loader, config)

    print("Fine-tuning completo...")
    unfreeze_model(model)
    train(model, train_loader, val_loader, config)

    print("Avaliação final...")
    metrics = test(model, test_loader, config)

    print(metrics)

if __name__ == "__main__":
    main()