import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Gabungkan data dari berbagai sumber (misalnya, beberapa file CSV)
data1 = pd.read_csv('data_gesture/halo_gesture_data.csv')
data2 = pd.read_csv('data_gesture/oke_gesture_data.csv')
data3 = pd.read_csv('data_gesture/i love u_gesture_data.csv')
data4 = pd.read_csv('data_gesture/well_gesture_data.csv')
data5 = pd.read_csv('data_gesture/A_gesture_data.csv')
data6 = pd.read_csv('data_gesture/Nama_gesture_data.csv')

data = pd.concat([data1, data2, data3, data4, data5, data6], ignore_index=True)

# Pastikan data sudah benar, periksa kolom label dan fitur
print(data.head())

# Pisahkan fitur dan label
X = data.drop('label', axis=1)  # Fitur adalah semua kolom selain label
y = data['label']  # Label adalah kolom 'label'

# Bagi data menjadi pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Simpan model yang telah dilatih
model_path = 'hasil_pelatihan/gesture_recognition_model.joblib'
joblib.dump(model, model_path)
print(f"Model berhasil disimpan di {model_path}")
