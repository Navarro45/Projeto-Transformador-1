from classes import EfficientNetClassifier


classifier = EfficientNetClassifier(
    train_dir="Modelo_2.0/data/train",
    test_dir="Modelo_2.0/data/test",
    epochs=5,
    batch_size=32
)

# treinar
classifier.train()

# avaliar
classifier.evaluate()

# salvar métricas
classifier.save_metrics()

# salvar modelo

# predição individual
classifier.predict_image(
    "Modelo_2.0/data/train/quasar/quasar_0.jpg"
)

# gradcam
classifier.generate_gradcam(
    "Modelo_2.0/data/train/quasar/quasar_0.jpg"
)
classifier.save_model()