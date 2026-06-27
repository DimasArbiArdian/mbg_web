import os

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import time
import html
import threading
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

from utils.predictor import NutritionPredictor


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_yolov8n_mbg_18kelas.pt"
)

RF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_gizi.pkl"
)

TKPI_PATH = os.path.join(
    BASE_DIR,
    "data",
    "data_tkpi.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sistem Deteksi MBG dan Klasifikasi Gizi",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .hero-box {
            background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 30px 32px;
            margin-bottom: 24px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.28);
        }

        .hero-title {
            font-size: 42px;
            font-weight: 850;
            color: #f8fafc;
            margin-bottom: 12px;
            letter-spacing: -0.8px;
            line-height: 1.15;
        }

        .hero-subtitle {
            font-size: 16px;
            color: #cbd5e1;
            line-height: 1.7;
            margin-bottom: 8px;
        }

        .hero-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .hero-badge {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 999px;
            padding: 8px 14px;
            color: #e5e7eb;
            font-size: 13px;
            font-weight: 650;
        }

        .section-title {
            font-size: 28px;
            font-weight: 780;
            color: #f8fafc;
            margin-top: 8px;
            margin-bottom: 16px;
            letter-spacing: -0.3px;
        }

        .section-subtitle {
            font-size: 15px;
            color: #94a3b8;
            margin-top: -8px;
            margin-bottom: 18px;
            line-height: 1.6;
        }

        .custom-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 20px 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.20);
            margin-bottom: 18px;
        }

        .upload-box {
            background: rgba(255,255,255,0.045);
            border: 1px dashed rgba(255,255,255,0.22);
            border-radius: 20px;
            padding: 20px 22px;
            margin-bottom: 18px;
            color: #cbd5e1;
            line-height: 1.7;
        }

        .info-box {
            background: rgba(59,130,246,0.10);
            border: 1px solid rgba(96,165,250,0.22);
            border-left: 6px solid #3b82f6;
            border-radius: 16px;
            padding: 16px 18px;
            color: #dbeafe;
            font-size: 15px;
            margin-bottom: 18px;
            line-height: 1.6;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(30,41,59,0.95) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 20px 18px;
            min-height: 132px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
        }

        .metric-label {
            font-size: 14px;
            color: #cbd5e1;
            margin-bottom: 10px;
            font-weight: 650;
        }

        .metric-value {
            font-size: 34px;
            font-weight: 850;
            color: #ffffff;
            line-height: 1.15;
            letter-spacing: -0.5px;
        }

        .metric-unit {
            font-size: 14px;
            color: #94a3b8;
            margin-top: 8px;
        }

        .status-baik {
            background: linear-gradient(135deg, #14532d 0%, #166534 100%);
            border: 1px solid rgba(34,197,94,0.28);
            border-left: 7px solid #22c55e;
            border-radius: 18px;
            padding: 20px;
            color: #ffffff;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 20px;
            box-shadow: 0 10px 26px rgba(20,83,45,0.32);
        }

        .status-cukup {
            background: linear-gradient(135deg, #713f12 0%, #854d0e 100%);
            border: 1px solid rgba(250,204,21,0.28);
            border-left: 7px solid #facc15;
            border-radius: 18px;
            padding: 20px;
            color: #ffffff;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 20px;
            box-shadow: 0 10px 26px rgba(113,63,18,0.30);
        }

        .status-kurang {
            background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
            border: 1px solid rgba(239,68,68,0.28);
            border-left: 7px solid #ef4444;
            border-radius: 18px;
            padding: 20px;
            color: #ffffff;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 20px;
            box-shadow: 0 10px 26px rgba(127,29,29,0.32);
        }

        .status-neutral {
            background: linear-gradient(135deg, #334155 0%, #475569 100%);
            border: 1px solid rgba(148,163,184,0.25);
            border-left: 7px solid #94a3b8;
            border-radius: 18px;
            padding: 20px;
            color: #ffffff;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 20px;
        }

        .detected-chip-box {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px 18px;
            color: #e5e7eb;
            margin-bottom: 18px;
            line-height: 1.7;
        }

        .small-label {
            color: #94a3b8;
            font-size: 13px;
            margin-bottom: 4px;
            font-weight: 650;
        }

        .footer-box {
            margin-top: 46px;
            text-align: center;
            color: #94a3b8;
            font-size: 14px;
            padding: 24px 12px;
            border-top: 1px solid rgba(255,255,255,0.08);
            line-height: 1.7;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .stButton>button,
        .stDownloadButton>button {
            border-radius: 14px;
            font-weight: 700;
            padding: 0.65rem 1rem;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px 14px 0 0;
            padding-left: 18px;
            padding-right: 18px;
            font-weight: 700;
        }

        .stFileUploader {
            background: rgba(255,255,255,0.02);
            border-radius: 16px;
        }

        img {
            border-radius: 16px;
        }

        hr {
            border-color: rgba(255,255,255,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_text(value):
    return html.escape(str(value))


def fmt_number(value, digits=2):
    try:
        number = float(value)
        if number.is_integer():
            return f"{number:.0f}"
        return f"{number:.{digits}f}"
    except Exception:
        return str(value)


def section_title(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{safe_text(title)}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{safe_text(subtitle)}</div>',
            unsafe_allow_html=True
        )


def metric_card(label, value, unit=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(value)}</div>
            <div class="metric-unit">{safe_text(unit)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def status_card(kategori):
    kategori_raw = str(kategori).strip()

    if kategori_raw == "Baik":
        css_class = "status-baik"
        icon = "✅"
        description = "Menu terdeteksi memiliki kecukupan gizi yang baik."
    elif kategori_raw == "Cukup":
        css_class = "status-cukup"
        icon = "⚠️"
        description = "Menu terdeteksi memiliki kecukupan gizi pada kategori cukup."
    elif kategori_raw == "Kurang":
        css_class = "status-kurang"
        icon = "❌"
        description = "Menu terdeteksi belum memenuhi kecukupan gizi."
    else:
        css_class = "status-neutral"
        icon = "ℹ️"
        description = "Kategori belum dapat ditentukan."

    st.markdown(
        f"""
        <div class="{css_class}">
            {icon} Kategori Kecukupan Gizi: {safe_text(kategori_raw)}
            <div style="font-size:14px;font-weight:500;margin-top:8px;opacity:0.88;">
                {safe_text(description)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def rename_df_item(df_item):
    if df_item is None or len(df_item) == 0:
        return pd.DataFrame()

    df = df_item.copy()

    rename_map = {
        "label": "Makanan",
        "confidence": "Confidence",
        "x1": "X1",
        "y1": "Y1",
        "x2": "X2",
        "y2": "Y2",
        "luas_bbox_norm": "Luas BBox Norm",
        "estimasi_gramasi": "Estimasi Gramasi (g)",
        "energi": "Energi (kkal)",
        "protein": "Protein (g)",
        "lemak": "Lemak (g)",
        "karbohidrat": "Karbohidrat (g)",
    }

    df = df.rename(columns=rename_map)

    preferred_columns = [
        "Makanan",
        "Confidence",
        "Estimasi Gramasi (g)",
        "Energi (kkal)",
        "Protein (g)",
        "Lemak (g)",
        "Karbohidrat (g)",
        "Luas BBox Norm",
        "X1",
        "Y1",
        "X2",
        "Y2",
    ]

    existing_columns = [col for col in preferred_columns if col in df.columns]
    df = df[existing_columns]

    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].round(2)

    return df


def rename_df_total(df_total):
    if df_total is None or len(df_total) == 0:
        return pd.DataFrame()

    df = df_total.copy()

    rename_map = {
        "total_gramasi": "Total Gramasi (g)",
        "total_energi": "Energi (kkal)",
        "total_protein": "Protein (g)",
        "total_lemak": "Lemak (g)",
        "total_karbohidrat": "Karbohidrat (g)",
        "prediksi_gizi": "Prediksi Gizi",
    }

    df = df.rename(columns=rename_map)

    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].round(2)

    return df


@st.cache_resource
def load_predictor(conf, imgsz):
    return NutritionPredictor(
        yolo_model_path=YOLO_MODEL_PATH,
        rf_model_path=RF_MODEL_PATH,
        tkpi_path=TKPI_PATH,
        conf=conf,
        imgsz=imgsz,
        batas_energi_baik=495
    )


def check_required_files():
    required_files = {
        "Model YOLOv8": YOLO_MODEL_PATH,
        "Model Random Forest": RF_MODEL_PATH,
        "Data TKPI": TKPI_PATH,
    }

    missing_files = []

    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append((name, path))

    return missing_files


def show_total_result(df_total):
    if df_total is None or len(df_total) == 0:
        st.warning("Hasil total nutrisi belum tersedia.")
        return

    row = df_total.iloc[0]
    kategori = row.get("prediksi_gizi", "Tidak diketahui")

    status_card(kategori)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card(
            "🍽️ Total Gramasi",
            fmt_number(row.get("total_gramasi", 0)),
            "gram"
        )

    with col2:
        metric_card(
            "🔥 Energi",
            fmt_number(row.get("total_energi", 0)),
            "kkal"
        )

    with col3:
        metric_card(
            "🥩 Protein",
            fmt_number(row.get("total_protein", 0)),
            "gram"
        )

    with col4:
        metric_card(
            "🧈 Lemak",
            fmt_number(row.get("total_lemak", 0)),
            "gram"
        )

    with col5:
        metric_card(
            "🍚 Karbohidrat",
            fmt_number(row.get("total_karbohidrat", 0)),
            "gram"
        )


def show_detected_summary(df_item, processing_time=None):
    if df_item is None or len(df_item) == 0:
        st.markdown(
            """
            <div class="detected-chip-box">
                <b>Objek Terdeteksi:</b> Tidak ada makanan yang terdeteksi.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    labels = df_item["label"].tolist() if "label" in df_item.columns else []
    labels_text = ", ".join([safe_text(label) for label in labels])

    if processing_time is not None:
        time_text = f"{processing_time:.2f} detik"
    else:
        time_text = "-"

    st.markdown(
        f"""
        <div class="detected-chip-box">
            <div class="small-label">Ringkasan Deteksi</div>
            <b>Objek Terdeteksi:</b> {labels_text}<br>
            <b>Jumlah Jenis Makanan:</b> {len(labels)}<br>
            <b>Waktu Pemrosesan:</b> {time_text}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">
                🍱 Sistem Deteksi Menu MBG dan Klasifikasi Kecukupan Gizi
            </div>
            <div class="hero-subtitle">
                Aplikasi ini digunakan untuk mendeteksi makanan pada menu
                <i>Makan Bergizi Gratis</i> menggunakan YOLOv8, menghitung
                estimasi kandungan nutrisi berdasarkan TKPI, dan mengklasifikasikan
                kecukupan gizi menggunakan Random Forest.
            </div>
            <div class="hero-badge-row">
                <div class="hero-badge">YOLOv8n</div>
                <div class="hero-badge">TKPI</div>
                <div class="hero-badge">Random Forest</div>
                <div class="hero-badge">18 Kelas Makanan</div>
                <div class="hero-badge">Prototype Skripsi</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan Model")
        st.caption("Atur parameter deteksi sesuai kebutuhan.")

        conf = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Semakin tinggi nilai confidence, semakin ketat deteksi objek."
        )

        imgsz = st.selectbox(
            "Image Size YOLO",
            options=[320, 416, 512, 640],
            index=1,
            help="Ukuran input YOLO. Nilai lebih besar dapat meningkatkan detail, tetapi lebih berat."
        )

        process_every_n_frames = st.slider(
            "Proses setiap N frame",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Digunakan untuk mode kamera real-time agar pemrosesan lebih ringan."
        )

        st.markdown("---")

        st.markdown(
            """
            **Tentang Sistem**

            - YOLOv8n untuk deteksi makanan
            - TKPI untuk perhitungan nutrisi
            - Random Forest untuk klasifikasi gizi
            - Estimasi gramasi berbasis bounding box
            """
        )

        st.markdown("---")

        st.caption(
            "Catatan: estimasi gramasi bersifat pendekatan komputasional, "
            "bukan pengganti hasil penimbangan fisik."
        )

    return conf, imgsz, process_every_n_frames


def render_upload_page(predictor):
    section_title(
        "📤 Deteksi dan Klasifikasi dari Gambar",
        "Upload gambar makanan, lalu sistem akan menampilkan hasil deteksi, estimasi nutrisi, dan kategori kecukupan gizi."
    )

    st.markdown(
        """
        <div class="upload-box">
            <b>Petunjuk penggunaan:</b><br>
            1. Upload gambar makanan dalam format JPG, JPEG, PNG, atau WEBP.<br>
            2. Sistem akan mendeteksi objek makanan menggunakan YOLOv8.<br>
            3. Sistem menghitung estimasi gramasi dan kandungan nutrisi berdasarkan TKPI.<br>
            4. Random Forest digunakan untuk menentukan kategori kecukupan gizi.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload gambar makanan",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="info-box">
                Silakan upload gambar makanan terlebih dahulu. Hasil deteksi dan klasifikasi akan muncul setelah gambar diproses.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Memproses gambar makanan..."):
            start_time = time.time()
            annotated_bgr, df_item, df_total = predictor.analyze_array(image_bgr)
            processing_time = time.time() - start_time

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        section_title("🖼️ Perbandingan Gambar")

        col_original, col_result = st.columns(2)

        with col_original:
            st.markdown('<div class="small-label">Gambar Asli</div>', unsafe_allow_html=True)
            st.image(
                image_rgb,
                caption="Gambar Asli",
                use_container_width=True
            )

        with col_result:
            st.markdown('<div class="small-label">Hasil Deteksi YOLOv8</div>', unsafe_allow_html=True)
            st.image(
                annotated_rgb,
                caption="Hasil Deteksi YOLOv8",
                use_container_width=True
            )

        st.markdown("---")

        hasil_tab1, hasil_tab2, hasil_tab3 = st.tabs(
            [
                "📊 Ringkasan Hasil",
                "🍽️ Detail Makanan",
                "⬇️ Download Data"
            ]
        )

        with hasil_tab1:
            section_title("📊 Hasil Klasifikasi Gizi")
            show_detected_summary(df_item, processing_time)
            show_total_result(df_total)

            st.markdown("---")
            section_title("📋 Total Nutrisi")
            st.dataframe(
                rename_df_total(df_total),
                use_container_width=True,
                hide_index=True
            )

        with hasil_tab2:
            section_title("🍽️ Detail Makanan Terdeteksi")

            if df_item is None or len(df_item) == 0:
                st.warning("Tidak ada makanan yang terdeteksi pada gambar.")
            else:
                st.dataframe(
                    rename_df_item(df_item),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")
            section_title("📋 Data Total Nutrisi")
            st.dataframe(
                rename_df_total(df_total),
                use_container_width=True,
                hide_index=True
            )

        with hasil_tab3:
            section_title("⬇️ Download Hasil Pengujian")

            base_filename = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")

            csv_item = df_item.to_csv(index=False).encode("utf-8")
            csv_total = df_total.to_csv(index=False).encode("utf-8")

            success, encoded_image = cv2.imencode(".png", annotated_bgr)
            image_bytes = encoded_image.tobytes() if success else None

            col_download1, col_download2, col_download3 = st.columns(3)

            with col_download1:
                st.download_button(
                    "⬇️ Detail Item CSV",
                    data=csv_item,
                    file_name=f"{base_filename}_detail_item.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_download2:
                st.download_button(
                    "⬇️ Total Nutrisi CSV",
                    data=csv_total,
                    file_name=f"{base_filename}_total_nutrisi.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_download3:
                if image_bytes is not None:
                    st.download_button(
                        "⬇️ Gambar Deteksi PNG",
                        data=image_bytes,
                        file_name=f"{base_filename}_hasil_deteksi.png",
                        mime="image/png",
                        use_container_width=True
                    )

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses gambar.")
        st.exception(e)


# =========================================================
# REAL-TIME CAMERA SECTION
# =========================================================

if WEBRTC_AVAILABLE:
    class RealTimeVideoProcessor(VideoProcessorBase):
        def __init__(self, predictor, process_every_n_frames=3):
            self.predictor = predictor
            self.process_every_n_frames = process_every_n_frames

            self.frame_count = 0
            self.lock = threading.Lock()

            self.last_annotated_bgr = None
            self.last_df_item = pd.DataFrame()
            self.last_df_total = pd.DataFrame()
            self.last_fps = 0.0

        def recv(self, frame):
            image_bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % self.process_every_n_frames != 0:
                if self.last_annotated_bgr is not None:
                    return av.VideoFrame.from_ndarray(
                        self.last_annotated_bgr,
                        format="bgr24"
                    )

                return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")

            start_time = time.time()

            try:
                annotated_bgr, df_item, df_total = self.predictor.analyze_array(
                    image_bgr
                )

                elapsed_time = time.time() - start_time
                fps = 1 / elapsed_time if elapsed_time > 0 else 0.0

                cv2.putText(
                    annotated_bgr,
                    f"FPS: {fps:.2f}",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 0),
                    2
                )

                with self.lock:
                    self.last_annotated_bgr = annotated_bgr
                    self.last_df_item = df_item
                    self.last_df_total = df_total
                    self.last_fps = fps

                return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

            except Exception as e:
                cv2.putText(
                    image_bgr,
                    f"Error: {str(e)}",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


def render_realtime_page(predictor, process_every_n_frames):
    section_title(
        "📷 Kamera Real-Time",
        "Mode ini digunakan untuk menampilkan deteksi makanan secara langsung melalui kamera."
    )

    st.markdown(
        """
        <div class="info-box">
            Fitur kamera digunakan untuk demo deteksi berbasis frame secara langsung.
            Untuk hasil gizi yang lebih stabil, klik tombol <b>Ambil Hasil Frame Terakhir</b>
            ketika posisi makanan sudah terlihat jelas.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not WEBRTC_AVAILABLE:
        st.error(
            "streamlit-webrtc atau av belum tersedia. "
            "Install terlebih dahulu dengan: pip install streamlit-webrtc av"
        )
        return

    ctx = webrtc_streamer(
        key="mbg-realtime-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: RealTimeVideoProcessor(
            predictor=predictor,
            process_every_n_frames=process_every_n_frames
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )

    st.markdown("---")

    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        capture_clicked = st.button(
            "📌 Ambil Hasil Frame Terakhir",
            use_container_width=True
        )

    with col_info:
        st.caption(
            "Gunakan tombol ini setelah kamera berhasil mendeteksi makanan "
            "dan bounding box terlihat stabil."
        )

    if capture_clicked:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                df_item = ctx.video_processor.last_df_item.copy()
                df_total = ctx.video_processor.last_df_total.copy()
                fps = ctx.video_processor.last_fps

            st.markdown(
                f"""
                <div class="detected-chip-box">
                    <div class="small-label">Performa Kamera</div>
                    <b>FPS terakhir:</b> {fps:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

            section_title("📊 Hasil Klasifikasi Gizi")
            show_total_result(df_total)

            st.markdown("---")
            section_title("🍽️ Detail Makanan Terdeteksi")
            st.dataframe(
                rename_df_item(df_item),
                use_container_width=True,
                hide_index=True
            )

            section_title("📋 Total Nutrisi")
            st.dataframe(
                rename_df_total(df_total),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("Kamera belum aktif atau frame belum tersedia.")


# =========================================================
# MAIN APP
# =========================================================

def main():
    missing_files = check_required_files()

    if missing_files:
        st.error("Beberapa file utama belum ditemukan.")
        for name, path in missing_files:
            st.write(f"- {name}: `{path}`")
        st.stop()

    conf, imgsz, process_every_n_frames = render_sidebar()

    predictor = load_predictor(conf, imgsz)

    render_header()

    tab_upload, tab_realtime = st.tabs(
        [
            "📤 Upload Gambar",
            "📷 Kamera Real-Time"
        ]
    )

    with tab_upload:
        render_upload_page(predictor)

    with tab_realtime:
        render_realtime_page(predictor, process_every_n_frames)

    st.markdown(
        """
        <div class="footer-box">
            Prototype Sistem Deteksi Menu MBG dan Klasifikasi Kecukupan Gizi<br>
            YOLOv8 • TKPI • Random Forest<br>
            Skripsi Dimas Arbi Ardian
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
