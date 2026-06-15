import os
import time
import threading

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from utils.predictor import NutritionPredictor


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


st.set_page_config(
    page_title="Sistem Deteksi MBG dan Klasifikasi Gizi",
    page_icon="🍱",
    layout="wide"
)


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


def show_total_result(df_total):
    if len(df_total) == 0:
        return

    row = df_total.iloc[0]
    kategori = row["prediksi_gizi"]

    if kategori == "Baik":
        st.success(f"Kategori Kecukupan Gizi: {kategori}")
    elif kategori == "Cukup":
        st.warning(f"Kategori Kecukupan Gizi: {kategori}")
    elif kategori == "Kurang":
        st.error(f"Kategori Kecukupan Gizi: {kategori}")
    else:
        st.info(f"Kategori Kecukupan Gizi: {kategori}")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Gramasi", f"{row['total_gramasi']} g")
    col2.metric("Energi", f"{row['total_energi']} kkal")
    col3.metric("Protein", f"{row['total_protein']} g")
    col4.metric("Lemak", f"{row['total_lemak']} g")
    col5.metric("Karbohidrat", f"{row['total_karbohidrat']} g")


class RealTimeVideoProcessor(VideoProcessorBase):
    def __init__(self, predictor, process_every_n_frames=3):
        self.predictor = predictor
        self.process_every_n_frames = process_every_n_frames

        self.frame_count = 0
        self.lock = threading.Lock()

        self.last_annotated_bgr = None
        self.last_df_item = pd.DataFrame()
        self.last_df_total = pd.DataFrame()
        self.last_fps = 0

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
            fps = 1 / elapsed_time if elapsed_time > 0 else 0

            cv2.putText(
                annotated_bgr,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
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
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


st.title("🍱 Sistem Deteksi Menu MBG dan Klasifikasi Kecukupan Gizi")
st.caption(
    "YOLOv8 untuk deteksi makanan, TKPI untuk perhitungan nutrisi, "
    "dan Random Forest untuk klasifikasi kecukupan gizi."
)

with st.sidebar:
    st.header("Pengaturan Model")

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    imgsz = st.selectbox(
        "Image Size YOLO",
        options=[320, 416, 512, 640],
        index=1
    )

    process_every_n_frames = st.slider(
        "Proses setiap N frame",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    st.caption(
        "Untuk kamera real-time, semakin kecil image size dan semakin besar "
        "nilai N frame, proses akan lebih ringan."
    )

predictor = load_predictor(conf, imgsz)

tab_upload, tab_realtime = st.tabs([
    "Upload Gambar",
    "Kamera Real-Time"
])


with tab_upload:
    st.subheader("Deteksi dan Klasifikasi dari Gambar")

    uploaded_file = st.file_uploader(
        "Upload gambar makanan",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Memproses gambar..."):
            annotated_bgr, df_item, df_total = predictor.analyze_array(
                image_bgr
            )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Gambar Asli", use_container_width=True)

        with col2:
            st.image(
                annotated_rgb,
                caption="Hasil Deteksi YOLOv8",
                use_container_width=True
            )

        st.subheader("Hasil Klasifikasi Gizi")
        show_total_result(df_total)

        st.subheader("Detail Makanan Terdeteksi")
        st.dataframe(df_item, use_container_width=True)

        st.subheader("Total Nutrisi")
        st.dataframe(df_total, use_container_width=True)

        csv_item = df_item.to_csv(index=False).encode("utf-8")
        csv_total = df_total.to_csv(index=False).encode("utf-8")

        col_download1, col_download2 = st.columns(2)

        with col_download1:
            st.download_button(
                "Download Detail Item CSV",
                data=csv_item,
                file_name="hasil_deteksi_item.csv",
                mime="text/csv"
            )

        with col_download2:
            st.download_button(
                "Download Total Nutrisi CSV",
                data=csv_total,
                file_name="hasil_total_nutrisi.csv",
                mime="text/csv"
            )


with tab_realtime:
    st.subheader("Deteksi Makanan Menggunakan Kamera Real-Time")

    st.info(
        "Mode ini menjalankan deteksi YOLOv8 pada frame kamera. "
        "Untuk hasil gizi yang lebih stabil, gunakan tombol "
        "'Ambil Hasil Frame Terakhir' setelah posisi makanan terlihat jelas."
    )

    ctx = webrtc_streamer(
        key="mbg-realtime-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: RealTimeVideoProcessor(
            predictor=predictor,
            process_every_n_frames=process_every_n_frames
        ),
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )

    st.divider()

    if st.button("Ambil Hasil Frame Terakhir"):
        if ctx.video_processor:
            with ctx.video_processor.lock:
                df_item = ctx.video_processor.last_df_item.copy()
                df_total = ctx.video_processor.last_df_total.copy()
                fps = ctx.video_processor.last_fps

            st.write(f"FPS terakhir: **{fps:.2f}**")

            st.subheader("Hasil Klasifikasi Gizi")
            show_total_result(df_total)

            st.subheader("Detail Makanan Terdeteksi")
            st.dataframe(df_item, use_container_width=True)

            st.subheader("Total Nutrisi")
            st.dataframe(df_total, use_container_width=True)

        else:
            st.warning("Kamera belum aktif atau frame belum tersedia.")