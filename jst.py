import numpy as np

# Fungsi aktivasi
def activation_function(y_in):
    return 1 if y_in >= 0 else -1

# Fungsi untuk melatih perceptron
def train_perceptron(X_train, y_train, learning_rate=0.1, epochs=100):
    num_samples, num_features = X_train.shape
    weights = np.random.rand(num_features + 1)  # Inisialisasi bobot dengan nilai acak kecil
    
    for epoch in range(epochs):
        for i in range(num_samples):
            x_with_bias = np.insert(X_train[i], 0, 1)  # Menambahkan nilai bias ke input
            y_in = np.dot(weights, x_with_bias)  # Menghitung nilai y_in
            y_pred = activation_function(y_in)  # Menggunakan fungsi aktivasi
            
            # Update bobot berdasarkan aturan perceptron
            weights += learning_rate * (y_train[i] - y_pred) * x_with_bias
    
    return weights

# Fungsi untuk melakukan prediksi dengan perceptron
def predict_perceptron(X, weights):
    num_samples = X.shape[0]
    predictions = np.zeros(num_samples)
    
    for i in range(num_samples):
        x_with_bias = np.insert(X[i], 0, 1)  # Menambahkan nilai bias ke input
        y_in = np.dot(weights, x_with_bias)  # Menghitung nilai y_in
        predictions[i] = activation_function(y_in)  # Menggunakan fungsi aktivasi
    
    return predictions

# Meminta input dari pengguna untuk jumlah data uji
num_user_inputs = int(input("Masukkan jumlah data uji yang ingin dimasukkan: "))

# Inisialisasi array untuk menyimpan input pengguna
user_inputs = []
targets = []

# Meminta input dari pengguna untuk setiap data uji
for i in range(num_user_inputs):
    user_input_X1 = float(input(f"Masukkan nilai X1 untuk data uji {i + 1}: "))
    user_input_X2 = float(input(f"Masukkan nilai X2 untuk data uji {i + 1}: "))
    target = int(input(f"Masukkan nilai target untuk data uji {i + 1} (1 atau -1): "))
    
    user_inputs.append([user_input_X1, user_input_X2])
    targets.append(target)

# Membuat data uji dari input pengguna
X_test_user_parabola = np.array(user_inputs)
y_test_user_parabola = np.array(targets)

# Melatih perceptron dengan dataset baru
trained_weights_parabola = train_perceptron(X_test_user_parabola, y_test_user_parabola, learning_rate=0.1, epochs=100)

# Melakukan prediksi pada data uji
y_pred_user_parabola = predict_perceptron(X_test_user_parabola, trained_weights_parabola)

# Mencetak prediksi
print("Prediksi Kelas untuk Data Uji Parabola:", y_pred_user_parabola)
