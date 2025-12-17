import datetime
import threading
import cv2
import math
import os
import time
import sys
import torch
import numpy as np

# PyTorch kütüphaneleri: RetinaNet MobileNetV3 kullanılıyor
from torchvision.models.detection import retinanet_mobilenet_v3_large  # 🚨 DAHA HIZLI MODEL
from torchvision.models.detection import RetinaNet_MobileNet_V3_Large_Weights  # Ağırlıklar
from torchvision.transforms import functional as F_T
from torchvision.transforms import Normalize
from torchvision.ops import nms
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- SABİT ÇÖZÜNÜRLÜK ---
RETINANET_INPUT_SIZE = 512  # Daha hafif model için giriş boyutu optimize edildi

# --- KRİTİK FİLTRE SABİTLERİ ---
MAX_BOX_AREA_RATIO = 0.90
MIN_BOX_PIXEL_AREA = 100
NMS_IOU_THRESHOLD = 0.7

# --- TAKİP PARAMETRELERİ (Optimum Ayarlar) ---
MAX_AGE = 60
HISTORY_LENGTH = 15
MOTION_THRESHOLD = 8

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
def load_retinanet_model():
    """PyTorch Hub üzerinden RetinaNet MobileNetV3 modelini yüklüyoruz."""
    print("[MODEL] RetinaNet MobileNetV3 modeli yükleniyor...")

    # Model yüklenir (weights='DEFAULT' ile)
    model = retinanet_mobilenet_v3_large(weights=RetinaNet_MobileNet_V3_Large_Weights.DEFAULT).eval()

    return model


# --- TESPİT FONKSİYONU ---
def get_detections(model, frame, confidence_threshold):
    """
    RetinaNet MobileNetV3 inference'ını çalıştırır ve DeepSORT uyumlu çıktı verir.
    """

    original_h, original_w, _ = frame.shape

    # 1. BGR -> RGB ve Resizing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Görüntü boyutu ayarlanır.
    resized_frame = cv2.resize(frame_rgb, (RETINANET_INPUT_SIZE, RETINANET_INPUT_SIZE))

    # 2. Ön İşleme
    input_tensor = F_T.to_tensor(resized_frame)

    # 3. Inference çalıştırma
    with torch.no_grad():
        prediction = model([input_tensor])[0]

        # 4. Çıktı Çözümleme ve Geri Ölçekleme
    results = []

    boxes = prediction['boxes'].cpu()
    labels = prediction['labels'].cpu()
    scores = prediction['scores'].cpu()

    # 4a. Güven Eşiği Filtresi
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # NMS Uygulaması
    if boxes.numel() > 0:
        nms_indices = nms(boxes, scores, NMS_IOU_THRESHOLD)
        boxes = boxes[nms_indices].numpy()
        labels = labels[nms_indices].numpy()
        scores = scores[nms_indices].numpy()
    else:
        return []

    # Ölçekleme faktörleri
    scale_x = original_w / RETINANET_INPUT_SIZE
    scale_y = original_h / RETINANET_INPUT_SIZE
    frame_area = original_w * original_h

    for box, label, score in zip(boxes, labels, scores):

        # Sadece Arka Plan (Class ID 0) olmayanları al
        if int(label) > 0:

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
                results.append([[xmin_orig, ymin_orig, width, height], float(score), int(label)])

    return results


def run_motion_tracking(video_path, confidence_threshold, output_filename, stop_event=threading.Event()):
    """
    RetinaNet MobileNetV3 Tabanlı Hareket analizli nesne takibi işlevi.
    """

    tracker = DeepSort(max_age=MAX_AGE)
    motion_history = {}

    # Video yakalama nesnesini başlat
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise RuntimeError(f"Video dosyası açılamadı veya bulunamadı: {video_path}")

    writer = create_video_writer(video_cap, output_filename)

    # Modeli yükle (RetinaNet MobileNetV3 çağrısı)
    try:
        model = load_retinanet_model()
    except Exception as e:
        video_cap.release()
        writer.release()
        raise RuntimeError(f"RetinaNet MobileNetV3 model yüklenirken hata: {e}")

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

        # --- KARE İŞLEME VE TESPİT (RetinaNet MobileNetV3) ---
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

            if len(motion_history[track_id]) > HISTORY_LENGTH:
                motion_history[track_id].pop(0)

            is_moving = False
            # HAREKET EŞİĞİ KONTROLÜ
            if len(motion_history[track_id]) == HISTORY_LENGTH:
                p_current = motion_history[track_id][-1]
                p_past = motion_history[track_id][0]
                distance = get_distance(p_current, p_past)
                if distance > MOTION_THRESHOLD:
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

        cv2.imshow(os.path.basename(video_path) + " (RetinaNet MobileNetV3)", frame)
        writer.write(frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            stop_event.set()
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()