# Import potřebných knihoven
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# Načtení a příprava dat
data_dir = 'data/raw'
train_file = f'{data_dir}/poker-hand-training.csv'
test_file = f'{data_dir}/poker-hand-testing.csv'

# Načtení trénovacích a testovacích dat
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Extrakce funkcí (features) a cílových hodnot (labels) z trénovacích dat
X_train = train_data.iloc[:, :-1].values  # Všechny sloupce kromě posledního (Poker Hand)
y_train = train_data.iloc[:, -1].values   # Poslední sloupec (Poker Hand)

# Extrakce funkcí (features) a cílových hodnot (labels) z testovacích dat
X_test = test_data.iloc[:, :-1].values  # Všechny sloupce kromě posledního (Poker Hand)
y_test = test_data.iloc[:, -1].values   # Poslední sloupec (Poker Hand)

print(f'Trénovací data: {X_train.shape}, {y_train.shape}')
print(f'Testovací data: {X_test.shape}, {y_test.shape}')

# Normalizace dat
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Převod na formát vhodný pro TensorFlow
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Změna rozměrů dat pro vstup do CNN (pokud budete používat CNN)
X_train = X_train.reshape(X_train.shape[0], 5, 2, 1)
X_test = X_test.reshape(X_test.shape[0], 5, 2, 1)

print(f'Počet trénovacích příkladů: {X_train.shape[0]}')
print(f'Počet testovacích příkladů: {X_test.shape[0]}')

# Definice modelu
model = models.Sequential([
    layers.Conv2D(32, (2, 2), activation='relu', input_shape=(5, 2, 1)),
    layers.MaxPooling2D((1, 1)),
    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.MaxPooling2D((1, 1)),
    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 tříd pro pokerové kombinace
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trénování modelu
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Uložení modelu
model.save('src/vision/poker_model.h5')
