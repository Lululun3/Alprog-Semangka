import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Muat model yang telah dilatih dari folder 'hasil_pelatihan'
model_path = 'hasil_pelatihan/gesture_recognition_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model berhasil dimuat.")
else:
    print(f"Model tidak ditemukan di {model_path}")
    exit()

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    result = hands.process(rgb_frame)

    # Jika tangan terdeteksi, ambil landmark dan prediksi gerakan
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ekstrak koordinat landmark tangan
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).flatten()  # Rata untuk masuk ke model

            # Prediksi gerakan menggunakan model
            prediction = model.predict([landmarks])
            gesture_name = prediction[0]

            # Tampilkan hasil prediksi di layar
            cv2.putText(frame, f'Gesture: {gesture_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Gesture Recognition", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
