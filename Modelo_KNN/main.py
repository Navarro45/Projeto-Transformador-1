from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from load_images import load_dataset
from data_output import data_output

train_path = "Modelo_2.0/data/train"
test_path = "Modelo_2.0/data/test"

X_train, y_train = load_dataset(train_path, max_per_class=3000)
X_test, y_test = load_dataset(test_path, max_per_class=3000)

# Criar modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Treinar
knn.fit(X_train, y_train)

# Prever
y_pred = knn.predict(X_test)

# Avaliar
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

data_output(knn, X_test, y_test)