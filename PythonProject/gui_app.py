import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import os
from datetime import datetime
import sys
import cv2  # cv2.destroyAllWindows() için

# --- İŞLEV İÇE AKTARMALARI (Çift Mod İçin Aliasing) ---

# --- MOD V2 (YOLOv8/YOLOv5) ---
try:
    from o_d_t_V2 import run_motion_tracking as run_v2_tracking
except ImportError as err:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] KRİTİK HATA: 'o_d_t_V2.py' içe aktarılamadı: {err}",
          file=sys.stderr)


    def run_v2_tracking(video_path, confidence_threshold, output_filename, stop_event):
        raise RuntimeError(f"YOLO Tabanlı V2 fonksiyonu yüklenemedi. Dosyayı kontrol edin. Orijinal hata: {err}")

# --- MOD V3 (SSD/CenterNet) ---
try:
    from o_d_t_V3 import run_motion_tracking as run_v3_tracking
except ImportError as err:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] KRİTİK HATA: 'o_d_t_V3.py' içe aktarılamadı: {err}",
          file=sys.stderr)


    def run_v3_tracking(video_path, confidence_threshold, output_filename, stop_event):
        raise RuntimeError(
            f"SSD/CenterNet Tabanlı V3 fonksiyonu yüklenemedi. Dosyayı kontrol edin. Orijinal hata: {err}")


# --- ANA GUI SINIFI ---

class ObjectTrackingGUI:
    """Nesne Algılama ve Takibi için Grafik Kullanıcı Arayüzü (GUI)."""

    def __init__(self, master):
        self.master = master
        master.title("Esnek Video İşleme (V2 & V3)")

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.video_path = tk.StringVar()
        # İki modu içeren yeni bir StringVar başlatıyoruz
        self.processing_mode = tk.StringVar(value="SSD/CenterNet Tabanlı Takip")
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.is_running = False
        self.process_thread = None
        self.stop_event = threading.Event()
        self.user_stopped = False

        # --- Arayüz Elemanları ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill='both', expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Kontroller", padding="10")
        control_frame.pack(side='left', fill='y', padx=10, pady=10)

        # Video Seçimi
        ttk.Label(control_frame, text="Video Dosyası:").grid(row=0, column=0, sticky="w", pady=5)
        self.video_entry = ttk.Entry(control_frame, textvariable=self.video_path, width=40)
        self.video_entry.grid(row=1, column=0, padx=5, sticky="ew")
        ttk.Button(control_frame, text="Seç", command=self.select_video).grid(row=1, column=1, padx=5)

        # 🎯 MOD SEÇİMİ (Yeniden Eklendi)
        ttk.Label(control_frame, text="İşleme Modu Seçimi:").grid(row=2, column=0, sticky="w", pady=5)
        self.mode_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.processing_mode,
            values=[
                "YOLO Tabanlı Takip (V2)",
                "SSD/CenterNet Tabanlı Takip (V3)"
            ],
            width=40,
            state="readonly"
        )
        self.mode_combobox.grid(row=3, column=0, columnspan=2, padx=5, sticky="ew")

        # Parametreler
        ttk.Label(control_frame, text="Güven Eşiği (0.0 - 1.0):").grid(row=4, column=0, sticky="w", pady=5)
        self.conf_scale = ttk.Scale(
            control_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, orient='horizontal', length=250,
            command=self.update_conf_label
        )
        self.conf_scale.grid(row=5, column=0, padx=5, sticky="ew")
        self.conf_label = ttk.Label(control_frame, text=f"{self.confidence_threshold.get():.2f}")
        self.conf_label.grid(row=5, column=1, padx=5)

        # Başlat/Durdur Düğmeleri
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=15)
        self.start_button = ttk.Button(button_frame, text="Başlat", command=self.start_processing, width=15)
        self.start_button.pack(side='left', padx=5)
        self.stop_button = ttk.Button(button_frame, text="Durdur", command=self.stop_processing, state=tk.DISABLED,
                                      width=15)
        self.stop_button.pack(side='left', padx=5)

        # Log Çerçevesi
        log_frame = ttk.LabelFrame(main_frame, text="Durum ve Loglar", padding="10")
        log_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, wrap='word', height=15, width=60, state=tk.DISABLED)
        self.log_text.pack(fill='both', expand=True)

        self.log_message("Uygulama hazır. Lütfen bir video dosyası seçin ve modunuzu belirleyin. 🎉")

    def update_conf_label(self, event):
        self.conf_label.config(text=f"{self.confidence_threshold.get():.2f}")

    def log_message(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def select_video(self):
        f_types = [('MP4 dosyaları', '*.mp4'), ('Tüm dosyalar', '*.*')]
        path = filedialog.askopenfilename(filetypes=f_types)
        if path:
            self.video_path.set(path)
            self.log_message(f"Video dosyası seçildi: {os.path.basename(path)}")

    def start_processing(self):
        video_path = self.video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Hata", "Lütfen önce geçerli bir video dosyası seçin.")
            return

        if self.is_running:
            return

        self.stop_event.clear()
        self.user_stopped = False
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        mode = self.processing_mode.get()
        self.log_message(f"İşleme başlatılıyor: {mode} (Eşik: {self.confidence_threshold.get():.2f})")

        self.process_thread = threading.Thread(target=self._run_video_processor, daemon=True)
        self.process_thread.start()

    def stop_processing(self):
        if not self.is_running:
            return

        self.stop_event.set()
        self.user_stopped = True
        self.log_message("İşleme durdurma isteği (Durdur Butonu) gönderildi. Lütfen işlemin sonlanmasını bekleyin.")

    def _run_video_processor(self):

        mode = self.processing_mode.get()
        video_path = self.video_path.get()
        conf_thresh = self.confidence_threshold.get()

        output_folder = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        processing_function = None
        output_type_name = ""

        # 🎯 DİNAMİK FONKSİYON SEÇİMİ
        if mode == "YOLO Tabanlı Takip (V2)":
            processing_function = run_v2_tracking
            output_type_name = "v2_yolo"
        elif mode == "SSD/CenterNet Tabanlı Takip (V3)":
            processing_function = run_v3_tracking
            output_type_name = "v3_ssd"

        output_filename = os.path.join(output_folder, f"output_{output_type_name}_{base_name}.mp4")

        try:
            self.log_message(f"Video penceresi açılıyor. Kapatmak için 'q' tuşunu kullanın.")
            self.log_message(f"Çıktı dosyası: {os.path.basename(output_filename)}")

            # Dinamik olarak seçilen fonksiyonu çağır
            processing_function(
                video_path=video_path,
                confidence_threshold=conf_thresh,
                output_filename=output_filename,
                stop_event=self.stop_event
            )

        except Exception as e:
            self.log_message(f"[KRİTİK HATA] İşlem sırasında bir hata oluştu: {e}")
            self.log_message("Lütfen model/video/bağımlılık ayarlarını kontrol edin.")
            self.stop_event.set()

        finally:
            self.is_running = False

            # Loglama
            if self.user_stopped:
                final_message = "İşlem Durdur Butonu ile sonlandırıldı. 🛑"
            elif self.stop_event.is_set():
                final_message = "İşlem 'q' tuşu veya beklenmedik bir durum nedeniyle sonlandırıldı. 🛑"
            else:
                final_message = "İşlem başarıyla tamamlandı (Video bitti). ✅"

            self.log_message(final_message)
            self.master.after(0, self._update_buttons_after_completion)
            cv2.destroyAllWindows()

    def _update_buttons_after_completion(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_closing(self):
        if self.is_running:
            self.stop_event.set()
            self.log_message("GUI kapatılıyor. İşlem sonlandırılıyor...")
        cv2.destroyAllWindows()
        self.master.destroy()


if __name__ == "__main__":
    try:
        cv2.destroyAllWindows()
        root = tk.Tk()
        app = ObjectTrackingGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"GUI başlatılırken hata oluştu: {e}")