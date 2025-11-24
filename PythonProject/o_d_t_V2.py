import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# --- PARAMETRELER VE EŞİKLER ---
# FPS Optimizasyonu için Algılama Eşiği (Eskiden 0.4'tü, şimdi artırıldı)
CONFIDENCE_THRESHOLD = 0.5
# Kısa süreli gizlenmeleri tolere etmek için Takip Toleransı
MAX_AGE = 70
# Duran/Hareketli Ayırımı için Merkez Noktası Kayıt Sayısı (N)
HISTORY_LENGTH = 9  # Son 15 kareden önceki pozisyonu kontrol et
# Nesnenin hareketli sayılması için minimum piksel mesafesi
MOTION_THRESHOLD = 3

# --- RENKLER ---
GREEN = (0, 255, 0)  # Hareketli nesne kutu rengi
RED = (0, 0, 255)  # Duran nesne kutu rengi
WHITE = (255, 255, 255)

# --- HAREKET TAKİP SÖZLÜĞÜ ---
# Format: {track_id: [{'center_x': x, 'center_y': y}, ...]}
motion_history = {}


# --- YARDIMCI FONKSİYON: MESAFE HESAPLAMA ---
def get_distance(p1, p2):
    """İki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# initialize the video capture object
# Kendi video dosya adınızı buraya yazın:
video_cap = cv2.VideoCapture("2.mp4")

# initialize the video writer object
writer = create_video_writer(video_cap, "output_optimized_motion_detection.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
# Geliştirilmiş max_age değeriyle DeepSORT'u başlat
tracker = DeepSort(max_age=MAX_AGE)

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # run the YOLO model on the frame
    detections = model(frame)[0]

    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        confidence = data[4]

        # FPS Optimizasyonu için yüksek güven eşiği ile filtrele
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # DeepSORT'a uygun formata ekle
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)

    # Mevcut karedeki takip kimliklerini topla
    current_track_ids = set()

    # loop over the tracks
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        current_track_ids.add(track_id)
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Nesnenin merkez noktasını hesapla
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        current_center = (center_x, center_y)

        # Nesnenin hareket geçmişini güncelle
        if track_id not in motion_history:
            motion_history[track_id] = []

        motion_history[track_id].append(current_center)

        # Geçmiş listesini HISTORY_LENGTH kadar sınırla
        if len(motion_history[track_id]) > HISTORY_LENGTH:
            motion_history[track_id].pop(0)

        # --- HAREKET KONTROL MANTIĞI ---
        is_moving = False

        # Yeterli geçmiş verisi varsa kontrol yap
        if len(motion_history[track_id]) == HISTORY_LENGTH:

            # Son pozisyon (current_center)
            p_current = motion_history[track_id][-1]
            # HISTORY_LENGTH kare önceki pozisyon
            p_past = motion_history[track_id][0]

            distance = get_distance(p_current, p_past)

            if distance > MOTION_THRESHOLD:
                is_moving = True

        # Kutu ve metin için renk ve etiket belirle
        box_color = GREEN if is_moving else RED
        motion_status = "HAREKETLI" if is_moving else "DURAN"

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.rectangle(frame, (xmin, ymin - 30), (xmin + 150, ymin), box_color, -1)

        # Üst kısımdaki etiketi çiz
        label = f"ID:{track_id} | {motion_status}"
        cv2.putText(frame, label, (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Nesnenin merkezini çiz
        cv2.circle(frame, current_center, 5, WHITE, -1)

    # Takip edilmeyen (kaybolan) nesneleri geçmiş listesinden temizle
    keys_to_delete = [tid for tid in motion_history if tid not in current_track_ids]
    for tid in keys_to_delete:
        del motion_history[tid]

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()