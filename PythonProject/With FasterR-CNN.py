import datetime
import threading
import cv2
import math
import os
import time
import sys
import torch
import numpy as np

# PyTorch kütüphaneleri: SSD yerine Faster R-CNN kullanılıyor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2  # 🚨 YENİ MODEL
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights  # Ağırlıklar
from torchvision.transforms import functional as F_T
from torchvision.transforms import Normalize
from torchvision.ops import nms  # Faster R-CNN kendi içinde NMS kullanır, ancak biz de kullanacağız
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- YENİ SABİT (Faster R-CNN için Giriş Boyutu esnektir, ancak biz stabilize edeceğiz) ---
FASTER_RCNN_INPUT_SIZE = 640  # Daha iyi sonuç için biraz daha büyük bir giriş boyutu

# --- KRİTİK FİLTRE SABİTLERİ ---
MAX_BOX_AREA_RATIO = 0.90
MIN_BOX_PIXEL_AREA = 100
NMS_IOU_THRESHOLD = 0.7

# --- TAKİP PARAMETRELERİ (SSD'den öğrendiğimiz optimum ayarlar) ---
MAX_AGE = 60  # Takip kaybı durumunda kutunun balonlaşma süresi kısaltıldı.
HISTORY_LENGTH = 15  # Hareketli/Duran teyidi için daha uzun geçmiş.
MOTION_THRESHOLD = 8  # Titreşim ve gürültüyü filtrelemek için eşik.

# --- SABİT NORMALİZASYON DEĞERLERİ ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NORMALIZE_TRANSFORM = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


# --- YARDIMCI FONKSİYON: MESAFE HESAPLAMA ---
def get_distance(p1, p2):
    """İki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# --- MODEL YÜKLEME ---
def load_frcnn_model():
    """PyTorch Hub üzerinden Faster R-CNN modelini yüklüyoruz."""
    print("[MODEL] Faster R-CNN ResNet50 FPN V2 modeli yükleniyor...")

    # Model yüklenir (weights='DEFAULT' ile)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

    return model


# --- TESPİT FONKSİYONU ---
def get_detections(model, frame, confidence_threshold):
    """
    Faster R-CNN inference'ını çalıştırır, ve DeepSORT uyumlu çıktı verir.
    """

    original_h, original_w, _ = frame.shape

    # 1. BGR -> RGB ve Resizing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🚨 Görüntü boyutu ayarlanır. Faster R-CNN esnektir ancak biz stabilize ediyoruz.
    resized_frame = cv2.resize(frame_rgb, (FASTER_RCNN_INPUT_SIZE, FASTER_RCNN_INPUT_SIZE))

    # 2. Ön İşleme (Sadece to_tensor kullanılır, BGR/RGB dönüşümü haricinde ekstra normalizasyon gereksiz)
    input_tensor = F_T.to_tensor(resized_frame)

    # 3. Inference çalıştırma
    with torch.no_grad():
        # Faster R-CNN kendi içinde NMS uygular ve sonuçları skor sırasına göre verir.
        prediction = model([input_tensor])[0]

        # 4. Çıktı Çözümleme ve Geri Ölçekleme
    results = []

    # PyTorch tahmin çıktıları
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # Ölçekleme faktörleri
    scale_x = original_w / FASTER_RCNN_INPUT_SIZE
    scale_y = original_h / FASTER_RCNN_INPUT_SIZE
    frame_area = original_w * original_h

    for box, label, score in zip(boxes, labels, scores):

        # Sadece eşiği geçenleri ve Arka Plan (Class ID 0) olmayanları al
        if score > confidence_threshold and int(label) > 0:

            # Kutuları orijinal video boyutuna ölçekle
            xmin_orig = int(box[0] * scale_x)
            ymin_orig = int(box[1] * scale_y)
            xmax_orig = int(box[2] * scale_x)
            ymax_orig = int(box[3] * scale_y)

            width = xmax_orig - xmin_orig
            height = ymax_orig - ymin_orig
            box_area = width * height

            # Gürültü ve Aşırı Büyük Kutu Filtreleri
            if box_area / frame_area > MAX_BOX_AREA_RATIO or box_area < MIN_BOX_PIXEL_AREA:
                continue

            if width > 0 and height > 0:
                # Faster R-CNN tespitleri, SSD'ye göre çok daha stabil olmalıdır
                results.append([[xmin_orig, ymin_orig, width, height], float(score), int(label)])

    return results


def run_motion_tracking(video_path, confidence_threshold, output_filename, stop_event=threading.Event()):
    """
    Faster R-CNN Tabanlı Hareket analizli nesne takibi işlevi.
    """

    tracker = DeepSort(max_age=MAX_AGE)  # MAX_AGE = 60
    motion_history = {}

    # Video yakalama nesnesini başlat
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise RuntimeError(f"Video dosyası açılamadı veya bulunamadı: {video_path}")

    writer = create_video_writer(video_cap, output_filename)

    # Modeli yükle (Faster R-CNN çağrısı)
    try:
        model = load_frcnn_model()
    except Exception as e:
        video_cap.release()
        writer.release()
        raise RuntimeError(f"Faster R-CNN model yüklenirken hata: {e}")

    frame_count = 0

    # --- ANA İŞLEME DÖNGÜSÜ ---
    while True:
        if stop_event.is_set():
            break

        start = datetime.datetime.now()
        ret, frame = video_cap.read()

        if not ret:
            stop_event.set()
            break

        frame_count += 1

        # --- KARE İŞLEME VE TESPİT (Faster R-CNN) ---
        results = get_detections(model, frame, confidence_threshold)

        # ------------------------------------------------
        # TRACKING VE HAREKET ANALİZİ
        # ------------------------------------------------

        tracks = tracker.update_tracks(results, frame=frame)
        current_track_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            current_track_ids.add(track_id)
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])

            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            current_center = (center_x, center_y)

            if track_id not in motion_history:
                motion_history[track_id] = []
            motion_history[track_id].append(current_center)

            if len(motion_history[track_id]) > HISTORY_LENGTH:  # HISTORY_LENGTH = 15
                motion_history[track_id].pop(0)

            is_moving = False
            # HAREKET EŞİĞİ KONTROLÜ
            if len(motion_history[track_id]) == HISTORY_LENGTH:
                p_current = motion_history[track_id][-1]
                p_past = motion_history[track_id][0]
                distance = get_distance(p_current, p_past)
                if distance > MOTION_THRESHOLD:  # MOTION_THRESHOLD = 8
                    is_moving = True

            box_color = GREEN if is_moving else RED
            motion_status = "HAREKETLI" if is_moving else "DURAN"

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
            cv2.rectangle(frame, (xmin, ymin - 30), (xmin + 150, ymin), box_color, -1)

            label = f"ID:{track_id} | {motion_status}"
            cv2.putText(frame, label, (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.circle(frame, current_center, 5, WHITE, -1)

        keys_to_delete = [tid for tid in motion_history if tid not in current_track_ids]
        for tid in keys_to_delete:
            del motion_history[tid]

        end = datetime.datetime.now()
        total_time = (end - start).total_seconds()
        fps_value = 1 / total_time if total_time > 0 else 0.0

        print(f"Frame: {frame_count}, Time: {total_time * 1000:.0f} ms, FPS: {fps_value:.2f}")

        fps_label = f"FPS: {fps_value:.2f}"
        cv2.putText(frame, fps_label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        cv2.imshow(os.path.basename(video_path) + " (Faster R-CNN)", frame)
        writer.write(frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            stop_event.set()
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()