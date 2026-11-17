"""
pipeline.py - pipeline ujung ke ujung klasifikasi penyakit daun cabai.

citra BGR (crop 1 daun) -> praproses -> segmentasi -> rekonstruksi -> fitur (22 + 4 bentuk) -> SVM

PENTING: konstanta di bagian KONFIGURASI harus sama dengan notebook 01-04.
Notebook 07 memeriksa konsistensinya terhadap output/fitur.csv.

Pemakaian dari terminal (di folder proyek):
    python pipeline.py foto_baru/daun1.jpg
    python pipeline.py foto_baru            # semua gambar di folder
"""
from pathlib import Path
import sys
import numpy as np
import cv2
import joblib
from scipy.ndimage import binary_fill_holes
from skimage.feature import graycomatrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
MODEL_PATH = OUT / "model" / "svm_final.joblib"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------- KONFIGURASI (sama dengan notebook 01-04) -----------------------------
SIZE = 512
GAUSS_K = (5, 5)
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
EQ_METHOD = "clahe"                      # "clahe" | "he"

ROI_FRAC = 0.40
OPEN_K, CLOSE_K = 7, 15
MIN_AREA_FRAC = 0.002
USE_SOBEL = False
HSV = {
    "hijau":   dict(aktif=True,  lo=(35, 40, 40),  hi=(85, 255, 255)),
    "kuning":  dict(aktif=True,  lo=(20, 60, 80),  hi=(35, 255, 255)),
    "cokelat": dict(aktif=False, lo=(8, 50, 20),   hi=(20, 255, 200)),
    "putih":   dict(aktif=False, lo=(0, 0, 200),   hi=(179, 40, 255)),
}

# batas peringatan kualitas (heuristik, bukan bagian pelatihan)
MASK_MIN_RATIO, MASK_MAX_RATIO = 0.05, 0.85

ANGLES = {0: 0.0, 45: np.pi / 4, 90: np.pi / 2, 135: 3 * np.pi / 4}
PROPS = ["ASM", "kontras", "IDM", "korelasi"]
COLOR_COLS = ["H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std"]
GLCM_COLS = [f"{p}_{a}" for a in ANGLES for p in PROPS]
SHAPE_COLS = ["luas", "keliling", "faktor_bentuk", "circularity"]
FEATURE_COLS = COLOR_COLS + GLCM_COLS + SHAPE_COLS


# ----------------------------- I/O -----------------------------
def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    buf.tofile(str(path))
    return ok


# ----------------------------- 01 praproses -----------------------------
def resize_keep_ratio(img, size=None):
    size = size or SIZE
    h, w = img.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def pad_black(img, size=None):
    size = size or SIZE
    h, w = img.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(img, top, size - h - top, left, size - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def equalize_L(img_bgr, method=None):
    method = method or EQ_METHOD
    L, A, B = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
    if method == "clahe":
        L2 = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(L)
    else:
        L2 = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


def preprocess(img):
    x = resize_keep_ratio(img)
    x = cv2.GaussianBlur(x, GAUSS_K, 0)
    x = equalize_L(x)
    return pad_black(x)


# ----------------------------- 02 segmentasi -----------------------------
def hsv_masks(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return {n: cv2.inRange(hsv, np.array(c["lo"]), np.array(c["hi"])) for n, c in HSV.items()}


def combine(masks):
    m = np.zeros_like(next(iter(masks.values())))
    for n, c in HSV.items():
        if c["aktif"]:
            m = cv2.bitwise_or(m, masks[n])
    return m


def sobel_edges(img_bgr):
    g = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def morph(m):
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)


def imfill(m):
    return (binary_fill_holes(m > 0) * 255).astype(np.uint8)


def roi_box(shape):
    h, w = shape[:2]
    bh, bw = int(h * ROI_FRAC), int(w * ROI_FRAC)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    return x0, y0, x0 + bw, y0 + bh


def filter_components(m):
    h, w = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    x0, y0, x1, y1 = roi_box(m.shape)
    touch_roi = set(np.unique(lab[y0:y1, x0:x1])) - {0}
    keep, kept = np.zeros_like(m), 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_FRAC * h * w and i in touch_roi:
            keep[lab == i] = 255
            kept += 1
    return keep, kept


def segment(img_bgr):
    m = combine(hsv_masks(img_bgr))
    if USE_SOBEL:
        m = cv2.bitwise_or(m, sobel_edges(img_bgr))
    m = imfill(morph(m))
    return filter_components(m)[0]


def overlay(img_bgr, mask, color=(0, 0, 255)):
    out = img_bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


# ----------------------------- 03 rekonstruksi -----------------------------
def reconstruct(img_bgr, mask):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


# ----------------------------- 04 fitur -----------------------------
def color_features(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[mask > 0].astype(np.float64)
    mu, sd = px.mean(0), px.std(0)
    return np.r_[mu[0], sd[0], mu[1], sd[1], mu[2], sd[2]]


def glcm_features(img_bgr, mask, distance=1):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask > 0)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    g, m = gray[sl].copy(), mask[sl] > 0
    g = np.clip(g, 1, 255)
    g[~m] = 0
    P = graycomatrix(g, [distance], list(ANGLES.values()), levels=256, symmetric=True, normed=False)
    P = P[:, :, 0, :].astype(np.float64)
    P[0, :, :] = 0
    P[:, 0, :] = 0
    s = P.sum(axis=(0, 1), keepdims=True)
    s[s == 0] = 1
    P /= s
    i = np.arange(256).reshape(-1, 1, 1)
    j = np.arange(256).reshape(1, -1, 1)
    asm = (P ** 2).sum(axis=(0, 1))
    con = (P * (i - j) ** 2).sum(axis=(0, 1))
    idm = (P / (1.0 + (i - j) ** 2)).sum(axis=(0, 1))
    mu_i, mu_j = (P * i).sum(axis=(0, 1)), (P * j).sum(axis=(0, 1))
    sd_i = np.sqrt((P * (i - mu_i) ** 2).sum(axis=(0, 1)))
    sd_j = np.sqrt((P * (j - mu_j) ** 2).sum(axis=(0, 1)))
    cor = np.where(sd_i * sd_j > 1e-12,
                   (P * (i - mu_i) * (j - mu_j)).sum(axis=(0, 1)) / np.maximum(sd_i * sd_j, 1e-12), 1.0)
    return np.stack([asm, con, idm, cor], axis=1).ravel()


def shape_features(mask):
    m = (mask > 0).astype(np.uint8)
    area = float(m.sum())
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts))
    form = 4 * np.pi * area / per ** 2 if per > 0 else 0.0
    (_, _), rad = cv2.minEnclosingCircle(np.vstack(cnts))
    circ = area / (np.pi * rad ** 2) if rad > 0 else 0.0
    return np.array([area, per, form, circ])


def extract_features(rec, mask):
    return np.r_[color_features(rec, mask), glcm_features(rec, mask), shape_features(mask)]


# ----------------------------- model + prediksi -----------------------------
def process_image(img_bgr):
    """Citra mentah -> dict (pre, mask, rec, fitur (26,) atau None, rasio_mask, status)."""
    pre = preprocess(img_bgr)
    mask = segment(pre)
    rec = reconstruct(pre, mask)
    ratio = float((mask > 0).mean())
    if not mask.any():
        return dict(pre=pre, mask=mask, rec=rec, fitur=None, rasio_mask=ratio, status="segmentasi_gagal")
    status = "ok" if MASK_MIN_RATIO <= ratio <= MASK_MAX_RATIO else "periksa_segmentasi"
    return dict(pre=pre, mask=mask, rec=rec, fitur=extract_features(rec, mask), rasio_mask=ratio, status=status)


def load_bundle(path=None):
    b = joblib.load(path or MODEL_PATH)
    b["id_to_name"] = {v: k for k, v in b["label_map"].items()}
    return b


def predict_features(bundle, fitur):
    """fitur: array 26 nilai (urut FEATURE_COLS) -> (nama_kelas, margin, skor_per_kelas)."""
    import pandas as pd
    X = pd.DataFrame([fitur], columns=FEATURE_COLS)[bundle["features"]]
    model = bundle["model"]
    pred_id = int(model.predict(X)[0])
    dec = np.atleast_1d(model.decision_function(X)[0])
    top2 = np.sort(dec)[::-1][:2]
    margin = float(top2[0] - top2[1]) if len(top2) > 1 else float(top2[0])
    skor = {bundle["id_to_name"][int(c)]: float(v) for c, v in zip(model.classes_, dec)}
    return bundle["id_to_name"][pred_id], margin, skor


def predict_image(img_bgr, bundle):
    res = process_image(img_bgr)
    if res["fitur"] is None:
        res.update(prediksi=None, margin=None, skor=None)
    else:
        res["prediksi"], res["margin"], res["skor"] = predict_features(bundle, res["fitur"])
    return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    files = sorted(p for p in target.iterdir() if p.suffix.lower() in IMG_EXT) if target.is_dir() else [target]
    bundle = load_bundle()
    for f in files:
        r = predict_image(imread(f), bundle)
        if r["prediksi"] is None:
            print(f"{f.name}: segmentasi gagal (mask kosong)")
        else:
            print(f"{f.name}: {r['prediksi']} | margin={r['margin']:.3f} | rasio mask={r['rasio_mask']:.2f} | {r['status']}")
