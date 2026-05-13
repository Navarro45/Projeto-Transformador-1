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

# avaliação detalhada dos testes
classifier.evaluate_test_results()

# salvar modelo
classifier.save_model()

# salvar dados do treino
classifier.save_training_data()

# predição individual
classifier.predict_image(
    "Modelo_2.0/data/train/quasar/quasar_0.jpg"
)

# gradcam
classifier.generate_gradcam(
    "Modelo_2.0/data/train/quasar/quasar_0.jpg"
)
classifier.save_model()