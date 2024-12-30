import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Fungsi untuk mendapatkan gambar tanda tangan dan label
def get_signatures_and_labels(main_path='/Volumes/DATA/Project_KB/UAS_TTD/ttd'):
    signatures = []
    labels = []
    label_names = {}
    current_label = 0

    # Iterasi melalui semua subfolder di folder utama
    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Membaca gambar
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  # Lewati jika gambar tidak valid
                
                # Resize gambar menjadi ukuran tetap (150x150)
                resized_img = cv2.resize(img, (150, 150))
                signatures.append(resized_img.flatten())  # Gambar diratakan untuk SVM
                labels.append(current_label)

            current_label += 1

    return np.array(signatures), np.array(labels), label_names

# Ambil gambar tanda tangan dan label
signatures, labels, label_names = get_signatures_and_labels()

# Membuat dan melatih model SVM
if len(signatures) > 0:
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(signatures, labels_encoded)

    # Simpan model pelatihan
    np.save('svm_model.npy', clf)
    np.save('label_encoder.npy', le)
else:
    print("Dataset kosong atau tidak valid. Pastikan dataset berisi gambar tanda tangan.")
    exit()

# Load model pelatihan yang telah disimpan
clf = np.load('svm_model.npy', allow_pickle=True).item()
le = np.load('label_encoder.npy', allow_pickle=True).item()

# Mulai kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses webcam. Pastikan webcam terhubung.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize frame menjadi ukuran tetap (150x150)
    resized_frame = cv2.resize(gray, (150, 150)).flatten().reshape(1, -1)

    # Prediksi tanda tangan
    label_encoded = clf.predict(resized_frame)

    # Mendapatkan probabilitas dari model SVM
    proba = clf.predict_proba(resized_frame)
    confidence = np.max(proba)

    # Decode label
    label = le.inverse_transform(label_encoded)[0]
    name = label_names.get(label, "Unknown")

    # Menampilkan hasil prediksi pada gambar
    cv2.putText(frame, f"{name} ({int(confidence * 100)}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame video
    cv2.imshow('Signature Recognition', frame)

    # Keluar jika tekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()