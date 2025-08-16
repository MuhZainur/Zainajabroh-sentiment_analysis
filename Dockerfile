# Dockerfile

# 1. Gunakan image dasar Python yang ringan
FROM python:3.9-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Buat pengguna non-root baru bernama "appuser"
#    -m -> buat home directory (/home/appuser)
#    -s /bin/bash -> set shell default
RUN useradd -m -s /bin/bash appuser

# 4. Salin file requirements terlebih dahulu
COPY ./requirements.txt /app/requirements.txt

# 5. Install semua library yang dibutuhkan
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 6. Salin sisa kode aplikasi Anda
#    --chown=appuser:appuser -> Ganti kepemilikan file menjadi milik appuser
COPY --chown=appuser:appuser ./main.py /app/main.py

# 7. Ganti pengguna dari root ke appuser
USER appuser

# 8. Buka port yang akan digunakan oleh aplikasi
#    PENTING: Hugging Face Spaces untuk Docker menggunakan port 7860
EXPOSE 7860

# 9. Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
