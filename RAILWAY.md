# Deploy worker ke Railway

Konfigurasi ini menjalankan `pipeline.py` sebagai worker privat. Worker memeriksa
bucket Supabase secara berkala, memprediksi gambar terbaru, mengunggah hasil
proses ke Supabase, dan mengirim hasil prediksi ke Firebase Realtime Database.
Worker ini bukan web server dan tidak memerlukan domain publik.

## Langkah deploy

1. Pastikan perubahan dan berkas model `output/model/svm_final.joblib` serta
   `output/rekomendasi_pengendalian.json` sudah ada di branch GitHub yang akan
   dihubungkan. Jangan commit `.env` atau secret.
2. Di Railway, buat project dari repository GitHub ini dan pilih branch tersebut.
   Railway akan membangun service menggunakan `Dockerfile` di root repository.
3. Tambahkan environment variables berikut di **Service → Variables**:

   | Variable | Wajib | Keterangan |
   | --- | --- | --- |
   | `SUPABASE_URL` | Ya | URL project Supabase |
   | `SUPABASE_KEY` | Ya | Key yang memiliki akses ke bucket sumber dan tujuan |
   | `SUPABASE_BUCKET_NAME` | Ya | Bucket yang berisi gambar masukan |
   | `FIREBASE_DATABASE_URL` | Ya* | URL Firebase Realtime Database |
   | `FIREBASE_PROJECT_ID` | Ya* | Alternatif untuk `FIREBASE_DATABASE_URL` |

   \*Atur salah satu dari `FIREBASE_DATABASE_URL` atau `FIREBASE_PROJECT_ID`.
   Jika Firebase Rules memerlukan autentikasi, atur juga `FIREBASE_AUTH_TOKEN`.

4. Atur **Restart Policy** ke `Always` dan jalankan satu replica saja agar worker
   tidak memproses gambar yang sama secara bersamaan.
5. Deploy, lalu periksa **Deploy Logs**. Saat berjalan, log menampilkan
   `Worker aktif` dan status pemeriksaan gambar.

Variable opsional:

- `SUPABASE_FOLDER`: folder di dalam bucket sumber; default root bucket.
- `SUPABASE_OUTPUT_BUCKET`: bucket tujuan untuk gambar preprocess, segmentasi,
  dan rekonstruksi; default `fusashield-results`. Pastikan bucket ini tersedia
  dan key memiliki izin upload.
- `SUPABASE_POLL_INTERVAL_SECONDS`: interval polling; default `10`.
- `FIREBASE_RESULT_PATH` dan `FIREBASE_HISTORY_PATH`: path Realtime Database;
  default `predictions/latest` dan `predictions/history`.

Worker hanya memilih gambar paling baru yang terlihat di bucket pada setiap
polling. Pastikan Storage Supabase dan Firebase Realtime Database dapat diakses
dengan credentials yang diberikan. Karena filesystem container bersifat
sementara, riwayat Firebase digunakan sebagai pemeriksaan duplikasi setelah
restart.
