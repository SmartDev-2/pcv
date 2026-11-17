"""
pipeline.py - pipeline ujung ke ujung klasifikasi penyakit daun cabai.

citra BGR (crop 1 daun) -> praproses -> segmentasi -> rekonstruksi -> fitur (22 + 4 bentuk) -> SVM

PENTING: konstanta di bagian KONFIGURASI harus sama dengan notebook 01-04.
Notebook 07 memeriksa konsistensinya terhadap output/fitur.csv.

Pemakaian dari terminal (di folder proyek):
    python pipeline.py foto_baru/daun1.jpg
    python pipeline.py foto_baru            # semua gambar di folder
    python pipeline.py                     # citra terbaru dari Supabase -> Firebase

Mode Supabase/Firebase membaca konfigurasi dari environment variable:
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET_NAME, SUPABASE_FOLDER
    FIREBASE_DATABASE_URL (atau FIREBASE_PROJECT_ID)
    FIREBASE_RESULT_PATH (opsional, default: predictions/latest)
    FIREBASE_HISTORY_PATH (opsional, default: predictions/history)
    FIREBASE_AUTH_TOKEN (opsional, jika rules Firebase memerlukannya)
    SUPABASE_POLL_INTERVAL_SECONDS (opsional, default: 10)
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import cv2
import joblib
from scipy.ndimage import binary_fill_holes
from skimage.feature import graycomatrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
MODEL_PATH = OUT / "model" / "svm_final.joblib"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SUPABASE_URL_ENV = "SUPABASE_URL"
SUPABASE_KEY_ENV = "SUPABASE_KEY"
SUPABASE_BUCKET_ENV = "SUPABASE_BUCKET_NAME"
SUPABASE_FOLDER_ENV = "SUPABASE_FOLDER"
SUPABASE_OUTPUT_BUCKET_ENV = "SUPABASE_OUTPUT_BUCKET"
FIREBASE_DATABASE_URL_ENV = "FIREBASE_DATABASE_URL"
FIREBASE_PROJECT_ID_ENV = "FIREBASE_PROJECT_ID"
FIREBASE_RESULT_PATH_ENV = "FIREBASE_RESULT_PATH"
FIREBASE_HISTORY_PATH_ENV = "FIREBASE_HISTORY_PATH"
FIREBASE_AUTH_TOKEN_ENV = "FIREBASE_AUTH_TOKEN"
RECOMMENDATIONS_PATH = OUT / "rekomendasi_pengendalian.json"
PROCESSED_STATE_PATH = OUT / ".processed_images.json"
MARGIN_RENDAH = 0.3

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
def _load_env_file(path=ROOT / ".env"):
    if not path.is_file():
        return
    assignment = re.compile(r"^\s*(?:\$env:)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = assignment.match(line)
        if not match:
            continue
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[match.group(1)] = value


_load_env_file()


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    buf.tofile(str(path))
    return ok


def _required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} belum diatur.")
    if value.startswith("<") and value.endswith(">"):
        raise RuntimeError(f"Environment variable {name} masih berisi placeholder.")
    return value


def _http_json(url, method="GET", body=None, headers=None):
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with _open_url(request, timeout=30) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} dari {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gagal mengakses {url}: {exc.reason}") from exc
    try:
        return json.loads(response_body.decode("utf-8")) if response_body else None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Respons bukan JSON yang valid dari {url}.") from exc


def _open_url(request, timeout):
    last_error = None
    for attempt in range(3):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError:
            raise
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_error


def _supabase_config():
    return {
        "url": _required_env(SUPABASE_URL_ENV).rstrip("/"),
        "key": _required_env(SUPABASE_KEY_ENV),
        "bucket": _required_env(SUPABASE_BUCKET_ENV),
        "folder": os.getenv(SUPABASE_FOLDER_ENV, "").strip("/"),
    }


def _supabase_output_bucket():
    return os.getenv(SUPABASE_OUTPUT_BUCKET_ENV, "fusashield-results")


def _upload_supabase_image(image, marker, filename):
    config = _supabase_config()
    digest = hashlib.sha256(_marker_key(marker).encode("utf-8")).hexdigest()[:16]
    output_path = f"processed/{digest}/{filename}.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Gagal mengenkode hasil proses: {filename}.png")

    bucket = _supabase_output_bucket()
    upload_url = (
        f"{config['url']}/storage/v1/object/"
        f"{quote(bucket, safe='')}/{quote(output_path, safe='/')}"
    )
    request = Request(
        upload_url,
        data=encoded.tobytes(),
        headers={
            "apikey": config["key"],
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "image/png",
            "x-upsert": "true",
        },
        method="POST",
    )
    try:
        with _open_url(request, timeout=60):
            pass
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Gagal upload {filename}.png ke Supabase (HTTP {exc.code}): {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Gagal upload {filename}.png ke Supabase: {exc.reason}") from exc

    public_url = (
        f"{config['url']}/storage/v1/object/public/"
        f"{quote(bucket, safe='')}/{quote(output_path, safe='/')}"
    )
    return public_url


def upload_processed_images(result, marker):
    return {
        "preprocess": _upload_supabase_image(
            result["pre"], marker, "preprocess"
        ),
        "segmentation": _upload_supabase_image(
            result["mask"], marker, "segmentation"
        ),
        "reconstruction": _upload_supabase_image(
            result["rec"], marker, "reconstruction"
        ),
    }


def _latest_supabase_object():
    config = _supabase_config()
    list_url = (
        f"{config['url']}/storage/v1/object/list/"
        f"{quote(config['bucket'], safe='')}"
    )
    items = _http_json(
        list_url,
        method="POST",
        body={
            "prefix": config["folder"],
            "limit": 1000,
            "offset": 0,
            "sortBy": {"column": "created_at", "order": "desc"},
        },
        headers={
            "apikey": config["key"],
            "Authorization": f"Bearer {config['key']}",
        },
    )
    if not isinstance(items, list):
        raise RuntimeError("Format daftar file Supabase tidak sesuai.")

    image_items = [
        item for item in items
        if isinstance(item, dict)
        and Path(item.get("name", "")).suffix.lower() in IMG_EXT
    ]
    if not image_items:
        location = config["folder"] or "(root bucket)"
        raise RuntimeError(f"Tidak ada citra di Supabase {location}.")

    def item_timestamp(item):
        value = item.get("updated_at") or item.get("created_at") or ""
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)

    latest = max(image_items, key=item_timestamp)
    object_name = latest["name"]
    object_path = f"{config['folder']}/{object_name}" if config["folder"] else object_name
    download_url = (
        f"{config['url']}/storage/v1/object/"
        f"{quote(config['bucket'], safe='')}/{quote(object_path, safe='/')}"
    )
    return config, object_path, latest, download_url


def _download_supabase_image(config, object_path, download_url):
    request = Request(
        download_url,
        headers={
            "apikey": config["key"],
            "Authorization": f"Bearer {config['key']}",
        },
    )
    try:
        with _open_url(request, timeout=60) as response:
            image_bytes = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Gagal mengunduh citra Supabase (HTTP {exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gagal mengunduh citra Supabase: {exc.reason}") from exc

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"File Supabase bukan citra yang dapat dibaca: {object_path}")
    return image


def _firebase_database_url():
    database_url = os.getenv(FIREBASE_DATABASE_URL_ENV)
    if database_url:
        return database_url.rstrip("/")
    project_id = _required_env(FIREBASE_PROJECT_ID_ENV)
    return f"https://{project_id}-default-rtdb.firebaseio.com"


def load_recommendations(path=None):
    source = Path(path or RECOMMENDATIONS_PATH)
    if not source.is_file():
        raise RuntimeError(
            f"File rekomendasi tidak ditemukan: {source}. "
            "Jalankan notebook 07 terlebih dahulu untuk membuatnya."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Gagal membaca file rekomendasi: {source}") from exc
    recommendations = payload.get("panduan_per_kelas")
    if not isinstance(recommendations, dict):
        raise RuntimeError(f"Format panduan_per_kelas tidak valid di {source}.")
    return recommendations


def _firebase_url(path):
    firebase_url = f"{_firebase_database_url()}/{quote(path.strip('/'), safe='/')}.json"
    auth_token = os.getenv(FIREBASE_AUTH_TOKEN_ENV)
    if auth_token:
        firebase_url += f"?auth={quote(auth_token, safe='')}"
    return firebase_url


def _history_contains(marker):
    history_path = os.getenv(FIREBASE_HISTORY_PATH_ENV, "predictions/history")
    records = _http_json(_firebase_url(history_path))
    if records is None:
        return False
    if not isinstance(records, dict):
        raise RuntimeError("Format data history Firebase tidak valid.")
    object_path, timestamp = marker
    return any(
        isinstance(record, dict)
        and record.get("image_path") == object_path
        and record.get("image_updated_at") == timestamp
        for record in records.values()
    )


def _marker_key(marker):
    return json.dumps(list(marker), ensure_ascii=False, separators=(",", ":"))


def _load_processed_markers():
    if not PROCESSED_STATE_PATH.is_file():
        return set()
    try:
        values = json.loads(PROCESSED_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Gagal membaca state citra yang sudah diproses: {PROCESSED_STATE_PATH}"
        ) from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RuntimeError(f"Format state citra tidak valid: {PROCESSED_STATE_PATH}")
    return set(values)


def _save_processed_markers(markers):
    PROCESSED_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = PROCESSED_STATE_PATH.with_suffix(".tmp")
    try:
        temporary_path.write_text(
            json.dumps(sorted(markers), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(PROCESSED_STATE_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Gagal menyimpan state citra yang sudah diproses: {PROCESSED_STATE_PATH}"
        ) from exc


def publish_result(
    result,
    object_path,
    metadata,
    recommendations=None,
    processed_images=None,
):
    result_path = os.getenv(FIREBASE_RESULT_PATH_ENV, "predictions/latest").strip("/")
    history_path = os.getenv(FIREBASE_HISTORY_PATH_ENV, "predictions/history").strip("/")
    prediction = result.get("prediksi")
    margin = result.get("margin")
    if recommendations is None:
        recommendations = load_recommendations()
    payload = {
        "image_path": object_path,
        "image_updated_at": metadata.get("updated_at") or metadata.get("created_at"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "prediction": prediction,
        "margin": margin,
        "confidence": (
            None if margin is None
            else "yakin" if margin >= MARGIN_RENDAH else "kurang_yakin"
        ),
        "mask_ratio": result["rasio_mask"],
        "scores": result.get("skor"),
        "recommendation": recommendations.get(prediction) if prediction else None,
        "processed_images": processed_images,
    }
    history_response = _http_json(
        _firebase_url(history_path),
        method="POST",
        body=payload,
    )
    if not isinstance(history_response, dict) or not history_response.get("name"):
        raise RuntimeError("Firebase tidak mengembalikan ID untuk log riwayat.")
    payload["history_id"] = history_response["name"]
    _http_json(_firebase_url(result_path), method="PUT", body=payload)
    return payload


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


def _process_latest_cloud_image(last_marker=None, processed_markers=None):
    processed_markers = processed_markers if processed_markers is not None else set()
    config, object_path, metadata, download_url = _latest_supabase_object()
    marker = (
        object_path,
        metadata.get("updated_at")
        or metadata.get("created_at"),
    )
    marker_key = _marker_key(marker)
    if (
        marker == last_marker
        or marker_key in processed_markers
        or _history_contains(marker)
    ):
        processed_markers.add(marker_key)
        _save_processed_markers(processed_markers)
        return marker, False

    image = _download_supabase_image(config, object_path, download_url)
    result = predict_image(image, load_bundle())
    processed_images = upload_processed_images(result, marker)
    payload = publish_result(
        result,
        object_path,
        metadata,
        load_recommendations(),
        processed_images,
    )
    prediction = payload["prediction"] or payload["status"]
    print(
        f"{object_path}: {prediction} | "
        f"rasio mask={payload['mask_ratio']:.2f} | "
        f"Firebase diperbarui"
    )
    processed_markers.add(marker_key)
    _save_processed_markers(processed_markers)
    return marker, True


def run_cloud_worker():
    raw_interval = os.getenv("SUPABASE_POLL_INTERVAL_SECONDS", "10")
    try:
        interval = float(raw_interval)
    except ValueError as exc:
        raise RuntimeError("SUPABASE_POLL_INTERVAL_SECONDS harus berupa angka.") from exc
    if interval <= 0:
        raise RuntimeError("SUPABASE_POLL_INTERVAL_SECONDS harus lebih besar dari 0.")

    last_marker = None
    processed_markers = _load_processed_markers()
    print(f"Worker aktif. Memeriksa citra Supabase setiap {interval:g} detik.")
    while True:
        try:
            marker, processed = _process_latest_cloud_image(last_marker, processed_markers)
            if not processed:
                print("Belum ada citra baru.")
            else:
                last_marker = marker
        except RuntimeError as exc:
            print(f"Menunggu: {exc}", file=sys.stderr)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nWorker dihentikan.")
            return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        try:
            run_cloud_worker()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        target = Path(sys.argv[1])
        files = (
            sorted(p for p in target.iterdir() if p.suffix.lower() in IMG_EXT)
            if target.is_dir()
            else [target]
        )
        bundle = load_bundle()
        for f in files:
            image = imread(f)
            if image is None:
                print(f"{f.name}: file bukan citra yang dapat dibaca", file=sys.stderr)
                continue
            r = predict_image(image, bundle)
            if r["prediksi"] is None:
                print(f"{f.name}: segmentasi gagal (mask kosong)")
            else:
                print(
                    f"{f.name}: {r['prediksi']} | margin={r['margin']:.3f} | "
                    f"rasio mask={r['rasio_mask']:.2f} | {r['status']}"
                )
