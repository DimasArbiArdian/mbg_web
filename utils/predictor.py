import os
import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO


class NutritionPredictor:
    def __init__(
        self,
        yolo_model_path,
        rf_model_path,
        tkpi_path,
        conf=0.35,
        imgsz=416,
        batas_energi_baik=495
    ):
        self.yolo_model = YOLO(yolo_model_path)
        self.rf_model = joblib.load(rf_model_path)
        self.tkpi_df = pd.read_csv(tkpi_path)

        self.conf = conf
        self.imgsz = imgsz
        self.batas_energi_baik = batas_energi_baik

        self.required_columns = [
            "label",
            "energi_kkal",
            "protein_g",
            "lemak_g",
            "karbohidrat_g"
        ]

        self._validate_tkpi()
        self.nutrition_data = self._build_nutrition_data()

        self.calibration_data = {
            "nasi": {"luas_referensi_norm": 0.225, "berat_referensi": 150},
            "ayam": {"luas_referensi_norm": 0.227, "berat_referensi": 100},
            "tempe": {"luas_referensi_norm": 0.139, "berat_referensi": 60},
            "timun": {"luas_referensi_norm": 0.085, "berat_referensi": 40},
            "jeruk": {"luas_referensi_norm": 0.092, "berat_referensi": 80},
            "tahu": {"luas_referensi_norm": 0.105, "berat_referensi": 60},
            "telur": {"luas_referensi_norm": 0.095, "berat_referensi": 60},
            "cap_cai": {"luas_referensi_norm": 0.180, "berat_referensi": 100},
            "lengkeng": {"luas_referensi_norm": 0.070, "berat_referensi": 50},
            "selada": {"luas_referensi_norm": 0.130, "berat_referensi": 40},
            "anggur": {"luas_referensi_norm": 0.080, "berat_referensi": 50},
            "semangka": {"luas_referensi_norm": 0.135, "berat_referensi": 120},
            "pisang": {"luas_referensi_norm": 0.145, "berat_referensi": 100},
            "mie": {"luas_referensi_norm": 0.210, "berat_referensi": 150},
            "kacang_panjang": {"luas_referensi_norm": 0.100, "berat_referensi": 50},
            "kentang": {"luas_referensi_norm": 0.130, "berat_referensi": 100},
            "daging_sapi": {"luas_referensi_norm": 0.150, "berat_referensi": 80},
            "bakso": {"luas_referensi_norm": 0.090, "berat_referensi": 60},
        }

        self.gramasi_limit = {
            "nasi": (100, 250),
            "mie": (80, 220),
            "kentang": (50, 180),

            "ayam": (60, 150),
            "telur": (40, 80),
            "daging_sapi": (50, 130),
            "bakso": (30, 120),

            "tahu": (40, 120),
            "tempe": (40, 120),

            "cap_cai": (50, 180),
            "kacang_panjang": (30, 120),
            "selada": (20, 80),
            "timun": (20, 100),

            "jeruk": (50, 150),
            "lengkeng": (30, 100),
            "anggur": (30, 120),
            "semangka": (50, 200),
            "pisang": (50, 150),
        }

    def _validate_tkpi(self):
        missing_columns = [
            col for col in self.required_columns
            if col not in self.tkpi_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Kolom berikut tidak ditemukan pada data TKPI: {missing_columns}"
            )

    def _build_nutrition_data(self):
        nutrition_data = {}

        for _, row in self.tkpi_df.iterrows():
            nutrition_data[row["label"]] = {
                "energi": row["energi_kkal"],
                "protein": row["protein_g"],
                "lemak": row["lemak_g"],
                "karbohidrat": row["karbohidrat_g"],
            }

        return nutrition_data

    def estimasi_gramasi(self, label, luas_bbox, image_width, image_height):
        if label not in self.calibration_data:
            return 0

        luas_gambar = image_width * image_height
        luas_bbox_norm = luas_bbox / luas_gambar

        luas_ref_norm = self.calibration_data[label]["luas_referensi_norm"]
        berat_ref = self.calibration_data[label]["berat_referensi"]

        gramasi = (luas_bbox_norm / luas_ref_norm) * berat_ref

        if label in self.gramasi_limit:
            min_g, max_g = self.gramasi_limit[label]
            gramasi = max(min_g, min(gramasi, max_g))

        return gramasi

    def hitung_nutrisi(self, label, gramasi):
        if label not in self.nutrition_data:
            return {
                "energi": 0,
                "protein": 0,
                "lemak": 0,
                "karbohidrat": 0,
            }

        data = self.nutrition_data[label]

        return {
            "energi": (gramasi / 100) * data["energi"],
            "protein": (gramasi / 100) * data["protein"],
            "lemak": (gramasi / 100) * data["lemak"],
            "karbohidrat": (gramasi / 100) * data["karbohidrat"],
        }

    def filter_duplikasi_label(self, df_item):
        if len(df_item) == 0:
            return df_item

        return (
            df_item
            .sort_values(by="confidence", ascending=False)
            .drop_duplicates(subset=["label"], keep="first")
            .reset_index(drop=True)
        )

    def koreksi_prediksi_gizi(self, prediksi_gizi, total_energi):
        if prediksi_gizi == "Baik" and total_energi < self.batas_energi_baik:
            return "Cukup"

        return prediksi_gizi

    def analyze_array(self, image_bgr):
        results = self.yolo_model.predict(
            source=image_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )

        result = results[0]
        image_height, image_width = result.orig_shape

        hasil_deteksi = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = self.yolo_model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            luas_bbox = width * height
            luas_bbox_norm = luas_bbox / (image_width * image_height)

            gramasi = self.estimasi_gramasi(
                label,
                luas_bbox,
                image_width,
                image_height
            )

            nutrisi = self.hitung_nutrisi(label, gramasi)

            hasil_deteksi.append({
                "label": label,
                "confidence": round(confidence, 3),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "luas_bbox_norm": round(luas_bbox_norm, 4),
                "estimasi_gramasi": round(gramasi, 2),
                "energi": round(nutrisi["energi"], 2),
                "protein": round(nutrisi["protein"], 2),
                "lemak": round(nutrisi["lemak"], 2),
                "karbohidrat": round(nutrisi["karbohidrat"], 2),
            })

        df_item = pd.DataFrame(hasil_deteksi)

        if len(df_item) > 0:
            df_item = self.filter_duplikasi_label(df_item)

        annotated_bgr = result.plot()

        df_total = self._build_total_result(df_item)

        return annotated_bgr, df_item, df_total

    def _build_total_result(self, df_item):
        if len(df_item) == 0:
            return pd.DataFrame([{
                "total_gramasi": 0,
                "total_energi": 0,
                "total_protein": 0,
                "total_lemak": 0,
                "total_karbohidrat": 0,
                "prediksi_gizi": "Tidak terdeteksi",
            }])

        total_gramasi = df_item["estimasi_gramasi"].sum()
        total_energi = df_item["energi"].sum()
        total_protein = df_item["protein"].sum()
        total_lemak = df_item["lemak"].sum()
        total_karbohidrat = df_item["karbohidrat"].sum()

        fitur_rf = pd.DataFrame([{
            "total_energi": total_energi,
            "total_protein": total_protein,
            "total_lemak": total_lemak,
            "total_karbohidrat": total_karbohidrat,
        }])

        prediksi_gizi = self.rf_model.predict(fitur_rf)[0]
        prediksi_gizi = self.koreksi_prediksi_gizi(
            prediksi_gizi,
            total_energi
        )

        return pd.DataFrame([{
            "total_gramasi": round(total_gramasi, 2),
            "total_energi": round(total_energi, 2),
            "total_protein": round(total_protein, 2),
            "total_lemak": round(total_lemak, 2),
            "total_karbohidrat": round(total_karbohidrat, 2),
            "prediksi_gizi": prediksi_gizi,
        }])