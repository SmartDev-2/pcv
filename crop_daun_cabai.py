"""
Tool crop daun cabai per penyakit.

Cara pakai:
    pip install pillow
    python crop_daun_cabai.py                      -> akan muncul dialog pilih folder gambar
    python crop_daun_cabai.py folder_gambar        -> hasil ke folder_gambar_crop
    python crop_daun_cabai.py folder_gambar hasil  -> hasil ke folder "hasil"

Kontrol:
    Drag mouse     : gambar kotak di area daun
    1 / 2 / 3 / 4  : simpan kotak sebagai kelas (lihat KELAS di bawah)
    Z              : batalkan (hapus) crop terakhir pada gambar ini
    Esc            : batalkan kotak yang sedang digambar
    D / Panah kanan: gambar berikutnya
    A / Panah kiri : gambar sebelumnya
    Q              : keluar
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageOps, ImageTk

# ====== UBAH DI SINI KALAU KELAS BERUBAH ======
KELAS = {
    "1": "sehat",
    "2": "ulat",
    "3": "virus_gemini",
    "4": "bercak_daun",
}
WARNA = {
    "1": "#22c55e",
    "2": "#f59e0b",
    "3": "#ef4444",
    "4": "#3b82f6",
}
UKURAN_MIN = 20  # piksel (di gambar asli), crop lebih kecil dari ini diabaikan
EKSTENSI = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ==============================================


class CropApp:
    def __init__(self, root, folder_in, folder_out):
        self.root = root
        self.folder_in = folder_in
        self.folder_out = folder_out
        self.files = sorted(
            f for f in os.listdir(folder_in) if f.lower().endswith(EKSTENSI)
        )
        if not self.files:
            messagebox.showerror("Kosong", "Tidak ada gambar di folder itu.")
            root.destroy()
            return

        for nama in KELAS.values():
            os.makedirs(os.path.join(folder_out, nama), exist_ok=True)

        self.idx = 0
        self.saved = {}  # idx -> list of (path_file, kotak_tampilan, kunci_kelas)
        self.img = None
        self.scale = 1.0
        self.tkimg = None
        self.start = None
        self.sel = None

        self.label = tk.Label(root, anchor="w", font=("Segoe UI", 10), justify="left")
        self.label.pack(fill="x", padx=6, pady=4)
        self.canvas = tk.Canvas(root, cursor="cross", highlightthickness=0, bg="#222")
        self.canvas.pack(padx=6, pady=4)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        root.bind("<Key>", self.on_key)

        self.load_image()

    # ---------- muat & gambar ulang ----------
    def load_image(self):
        path = os.path.join(self.folder_in, self.files[self.idx])
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")  # perbaiki rotasi foto HP
        self.img = img

        maxw = int(self.root.winfo_screenwidth() * 0.9)
        maxh = int(self.root.winfo_screenheight() * 0.78)
        self.scale = min(maxw / img.width, maxh / img.height, 1.0)
        dw, dh = int(img.width * self.scale), int(img.height * self.scale)
        self.dw, self.dh = dw, dh
        self.tkimg = ImageTk.PhotoImage(img.resize((dw, dh)))
        self.canvas.config(width=dw, height=dh)
        self.sel = None
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)
        for _, (x0, y0, x1, y1), k in self.saved.get(self.idx, []):
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=WARNA[k], width=3)
            self.canvas.create_text(
                x0 + 4, y0 + 4, anchor="nw", text=KELAS[k],
                fill=WARNA[k], font=("Segoe UI", 11, "bold"),
            )
        if self.sel:
            self.canvas.create_rectangle(*self.sel, outline="white", dash=(4, 3), width=2)
        self.update_label()

    def update_label(self):
        legenda = "   ".join(f"[{k}] {v}" for k, v in KELAS.items())
        jumlah = len(self.saved.get(self.idx, []))
        self.label.config(
            text=(
                f"Gambar {self.idx + 1}/{len(self.files)}: {self.files[self.idx]}   |   "
                f"crop di gambar ini: {jumlah}\n"
                f"{legenda}   |   Z=undo  D=next  A=prev  Esc=batal kotak  Q=keluar"
            )
        )

    # ---------- mouse ----------
    def clamp(self, x, y):
        return max(0, min(x, self.dw)), max(0, min(y, self.dh))

    def on_press(self, e):
        self.start = self.clamp(e.x, e.y)
        self.sel = None

    def on_drag(self, e):
        if not self.start:
            return
        x, y = self.clamp(e.x, e.y)
        self.sel = (
            min(self.start[0], x), min(self.start[1], y),
            max(self.start[0], x), max(self.start[1], y),
        )
        self.redraw()

    def on_release(self, e):
        if self.sel and (self.sel[2] - self.sel[0] < 5 or self.sel[3] - self.sel[1] < 5):
            self.sel = None
        self.start = None
        self.redraw()

    # ---------- keyboard ----------
    def on_key(self, e):
        k = e.char
        ks = e.keysym
        if k in KELAS:
            self.save_crop(k)
        elif k.lower() == "z":
            self.undo()
        elif k.lower() == "d" or ks == "Right":
            self.goto(self.idx + 1)
        elif k.lower() == "a" or ks == "Left":
            self.goto(self.idx - 1)
        elif ks == "Escape":
            self.sel = None
            self.redraw()
        elif k.lower() == "q":
            self.root.destroy()

    def goto(self, i):
        if 0 <= i < len(self.files):
            self.idx = i
            self.load_image()

    # ---------- simpan & undo ----------
    def save_crop(self, kunci):
        if not self.sel:
            return
        x0, y0, x1, y1 = self.sel
        box = tuple(int(round(v / self.scale)) for v in (x0, y0, x1, y1))
        box = (
            max(0, box[0]), max(0, box[1]),
            min(self.img.width, box[2]), min(self.img.height, box[3]),
        )
        if box[2] - box[0] < UKURAN_MIN or box[3] - box[1] < UKURAN_MIN:
            self.sel = None
            self.redraw()
            return

        crop = self.img.crop(box)
        kelas = KELAS[kunci]
        stem = os.path.splitext(self.files[self.idx])[0]
        n = 1
        while True:
            out = os.path.join(self.folder_out, kelas, f"{stem}_{kelas}_{n:02d}.jpg")
            if not os.path.exists(out):
                break
            n += 1
        crop.save(out, quality=95)

        self.saved.setdefault(self.idx, []).append((out, self.sel, kunci))
        self.sel = None
        self.redraw()

    def undo(self):
        daftar = self.saved.get(self.idx, [])
        if not daftar:
            return
        path, _, _ = daftar.pop()
        if os.path.exists(path):
            os.remove(path)
        self.redraw()


def main():
    folder_in = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    root.title("Crop Daun Cabai per Penyakit")
    if not folder_in:
        folder_in = filedialog.askdirectory(title="Pilih folder gambar mentah")
        if not folder_in:
            return
    folder_out = (
        sys.argv[2] if len(sys.argv) > 2
        else folder_in.rstrip("/\\") + "_crop"
    )
    CropApp(root, folder_in, folder_out)
    root.mainloop()


if __name__ == "__main__":
    main()
