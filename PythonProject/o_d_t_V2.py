import datetime
import threading  # GUI'den durdurma kontrolü için
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import os

# --- SABİT PARAMETRELER (Genel tanım, fonksiyon içinde de kullanılacak) ---
MAX_AGE = 70
HISTORY_LENGTH = 9
MOTION_THRESHOLD = 3
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


# --- YARDIMCI FONKSİYON: MESAFE HESAPLAMA ---
def get_distance(p1, p2):
    """İki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def run_motion_tracking(video_path, confidence_threshold, output_filename, stop_event=threading.Event()):
    """
    Hareket analizli nesne takibi işlevi.
    GUI'den çağrılır ve video penceresini açar.
    """

    # Yeni bir DeepSORT nesnesi ve motion_history sözlüğü başlat
    tracker = DeepSort(max_age=MAX_AGE)
    motion_history = {}

    # Video yakalama nesnesini başlat
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"[HATA] Video dosyası açılamadı: {video_path}")
        return

    # Video yazma nesnesini başlat
    writer = create_video_writer(video_cap, output_filename)

    # Modeli yükle
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"[HATA] YOLO model yüklenemedi: {e}")
        video_cap.release()
        writer.release()
        return

    frame_count = 0

    # --- ANA İŞLEME DÖNGÜSÜ ---
    while True:
        # Harici durdurma sinyalini kontrol et
        if stop_event.is_set():
            break

        start = datetime.datetime.now()
        ret, frame = video_cap.read()

        if not ret:
            # Video bittiğinde durdurma sinyalini gönder ve döngüden çık
            stop_event.set()
            break

        frame_count += 1

        # --- KARE İŞLEME VE TAKİP MANTIĞI (Orijinal koddan taşındı) ---
        detections = model(frame)[0]
        results = []

        # DETECTION
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < confidence_threshold:
                continue
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # TRACKING
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

        # Konsola FPS ve işleme süresini yaz
        print(f"Frame: {frame_count}, Time: {total_time * 1000:.0f} ms, FPS: {fps_value:.2f}")

        # FPS'yi kare üzerine çiz
        fps_label = f"FPS: {fps_value:.2f}"
        cv2.putText(frame, fps_label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # --- GÖRÜNTÜLEME VE MANUEL DURDURMA KONTROLÜ (Tekrar eklendi) ---
        cv2.imshow(os.path.basename(video_path), frame)  # Pencere başlığı video adı olsun
        writer.write(frame)

        key = cv2.waitKey(1)

        # 'q' tuşuna basıldığında hem yerel döngüyü kır hem de GUI'ye sinyal gönder
        if key == ord("q"):
            stop_event.set()
            break

        # Video akışı bittiğinde sadece yerel döngüyü kır
        if not ret:
            break

    # --- KAYNAKLARI SERBEST BIRAK ---
    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

# Not: Modülerleştirilmiş kodun dışında kalan tüm sıralı çalıştırma kodlarını sildiğinizden emin olun!