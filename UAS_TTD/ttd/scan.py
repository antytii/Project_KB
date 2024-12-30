import cv2
import os
from datetime import datetime
import time

# Fungsi untuk menyimpan tanda tangan ke folder dataset
def save_signature(frame, name, save_dir="/Volumes/DATA/Project_KB/UAS_TTD/ttd"):
    if not os.path.exists(save_dir):  # Buat folder dataset jika belum ada
        os.makedirs(save_dir)
    person_dir = os.path.join(save_dir, name)  # Folder khusus untuk setiap orang
    if not os.path.exists(person_dir):  # Buat folder jika belum ada
        os.makedirs(person_dir)

    # Buat nama file unik menggunakan timestamp
    filename = os.path.join(person_dir, f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[INFO] Tanda tangan disimpan di: {filename}")

# Inisialisasi webcam
print("[INFO] Mulai menangkap gambar tanda tangan...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)  # Waktu pemanasan webcam

name = input("Masukkan nama untuk tanda tangan ini: ")  # Nama orang untuk folder dataset

# Loop untuk menangkap gambar
while True:
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Tidak dapat membaca dari webcam")
        break

    frame = cv2.resize(frame, (500, 500))  # Resize untuk konsistensi ukuran
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale untuk keseragaman

    # Menampilkan video di layar
    cv2.imshow("Capture Signatures - Tekan 's' untuk menyimpan, 'q' untuk keluar", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Simpan gambar jika tombol 's' ditekan
        save_signature(gray, name)
    elif key == ord('q'):  # Keluar jika tombol 'q' ditekan
        break

# Tutup proses
vs.release()
cv2.destroyAllWindows()
print("[INFO] Selesai menangkap gambar tanda tangan.")