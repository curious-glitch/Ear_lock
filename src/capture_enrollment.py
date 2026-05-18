import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import requests

# =========================================================
# CONFIG
# =========================================================
ROOT = Path(r"Z:\Code\EarEdge3")
ENROLL_DIR = ROOT / "enroll_data"

PERSON_NAME = "Anto"          # change this
USE_RIGHT_EAR = False         # False = left ear, True = right ear

# =========================================================
# CAMERA SOURCE
# =========================================================
# For DroidCam / virtual webcam, keep this True
USE_WEBCAM_INDEX = False

# Change this after testing camera index
# Usually:
# 0 = laptop webcam
# 1 or 2 = DroidCam
WEBCAM_INDEX = 2
WEBCAM_BACKEND = cv2.CAP_DSHOW   # good for Windows

# Phone stream fallback (only used if USE_WEBCAM_INDEX = False)
PHONE_IP = "172.30.10.243"
PORT = "4747"
USERNAME = "earedge"
PASSWORD = "19892"

USE_MJPEG = True
MJPEG_URL = f"http://{USERNAME}:{PASSWORD}@{PHONE_IP}:{PORT}/video"
SHOT_URL = f"http://{USERNAME}:{PASSWORD}@{PHONE_IP}:{PORT}/shot.jpg"

# =========================================================
# OUTPUT
# =========================================================
SAVE_SIZE = 224
TARGET_IMAGES = 200

# =========================================================
# VIEW
# =========================================================
WINDOW_W = 1000
WINDOW_H = 700

# Crop ratios
RIGHT_CROP = (0.58, 0.18, 0.92, 0.88)
LEFT_CROP  = (0.08, 0.18, 0.42, 0.88)

# =========================================================
# QUALITY THRESHOLDS
# =========================================================
MIN_SHARPNESS = 120.0
MIN_BRIGHTNESS = 55.0
MAX_BRIGHTNESS = 210.0
MAX_MOTION_DIFF = 8.0
MIN_CAPTURE_INTERVAL = 0.8
MIN_HASH_DISTANCE = 10

# =========================================================
# CAMERA READER
# =========================================================
class CameraReader:
    def __init__(self, use_webcam_index=False, webcam_index=0,
                 use_mjpeg=True, mjpeg_url=None, shot_url=None,
                 username=None, password=None, webcam_backend=cv2.CAP_DSHOW):
        self.use_webcam_index = use_webcam_index
        self.webcam_index = webcam_index
        self.use_mjpeg = use_mjpeg
        self.mjpeg_url = mjpeg_url
        self.shot_url = shot_url
        self.auth = (username, password)
        self.session = requests.Session()
        self.cap = None
        self.mode = None
        self.webcam_backend = webcam_backend

    def open(self):
        if self.use_webcam_index:
            self.cap = cv2.VideoCapture(self.webcam_index, self.webcam_backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open webcam index {self.webcam_index}")

            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Webcam index {self.webcam_index} opened but no frames received")

            self.mode = f"WEBCAM {self.webcam_index}"
            return

        if self.use_mjpeg:
            self.cap = cv2.VideoCapture(self.mjpeg_url, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.mode = "MJPEG"
                    return
                self.cap.release()
                self.cap = None

        self.mode = "SNAPSHOT"

    def read(self):
        if self.mode and (self.mode.startswith("WEBCAM") or self.mode == "MJPEG"):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            return None

        try:
            response = self.session.get(self.shot_url, auth=self.auth, timeout=3)
            if response.status_code != 200:
                return None
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def release(self):
        if self.cap is not None:
            self.cap.release()

# =========================================================
# HELPERS
# =========================================================
def compute_crop_box(frame, use_right_ear):
    h, w, _ = frame.shape
    rx1, ry1, rx2, ry2 = RIGHT_CROP if use_right_ear else LEFT_CROP
    x1, y1 = int(rx1 * w), int(ry1 * h)
    x2, y2 = int(rx2 * w), int(ry2 * h)
    return x1, y1, x2, y2

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def motion_diff(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    g1 = cv2.resize(g1, (64, 64))
    g2 = cv2.resize(g2, (64, 64))
    return float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))))

def average_hash(img, hash_size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size))
    mean = resized.mean()
    return (resized > mean).astype(np.uint8).flatten()

def hash_distance(h1, h2):
    return int(np.sum(h1 != h2))

def is_duplicate(current_crop, saved_hashes, min_hash_distance):
    if len(saved_hashes) == 0:
        return False, None

    h = average_hash(current_crop)
    distances = [hash_distance(h, sh) for sh in saved_hashes]
    best = min(distances)
    return best < min_hash_distance, h

def save_crop(crop, save_path):
    crop_resized = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE))
    cv2.imwrite(str(save_path), crop_resized)

def draw_status_panel(frame, person_name, saved_count, target_images,
                      status_text, status_color, sharpness, brightness,
                      motion, camera_mode, side_label):
    overlay = frame.copy()
    h, w, _ = frame.shape

    cv2.rectangle(overlay, (15, 15), (520, 190), (20, 20, 25), -1)
    cv2.rectangle(overlay, (w - 280, 15), (w - 15, 150), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    cv2.rectangle(frame, (25, 25), (500, 65), status_color, -1)

    cv2.putText(frame, status_text, (35, 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    cv2.putText(frame, f"Person: {person_name}", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Saved: {saved_count}/{target_images}", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 255, 180), 2)
    cv2.putText(frame, f"Sharpness: {sharpness:.1f}", (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
    cv2.putText(frame, f"Brightness: {brightness:.1f}   Motion: {motion:.1f}", (30, 182),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)

    cv2.putText(frame, "SYSTEM", (w - 255, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    cv2.putText(frame, f"Cam: {camera_mode}", (w - 255, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, f"Ear: {side_label}", (w - 255, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, "q quit   s save   e toggle ear", (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

# =========================================================
# MAIN
# =========================================================
def main():
    person_dir = ENROLL_DIR / PERSON_NAME
    person_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(person_dir.glob("*.jpg"))
    saved_count = len(existing_files)

    saved_hashes = []
    for f in existing_files:
        img = cv2.imread(str(f))
        if img is not None:
            saved_hashes.append(average_hash(img))

    prev_crop = None
    recent_motion = deque(maxlen=3)
    last_capture_time = 0

    use_right_ear = USE_RIGHT_EAR

    cam = CameraReader(
        use_webcam_index=USE_WEBCAM_INDEX,
        webcam_index=WEBCAM_INDEX,
        use_mjpeg=USE_MJPEG,
        mjpeg_url=MJPEG_URL,
        shot_url=SHOT_URL,
        username=USERNAME,
        password=PASSWORD,
        webcam_backend=WEBCAM_BACKEND
    )
    cam.open()

    print(f"Camera mode: {cam.mode}")
    print(f"Saving to: {person_dir}")
    print("Press 'q' to quit, 's' to manual save, 'e' to toggle ear side")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

        x1, y1, x2, y2 = compute_crop_box(frame, use_right_ear)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        sharpness = sharpness_score(crop)
        brightness = brightness_score(crop)

        if prev_crop is None:
            motion = 0.0
        else:
            motion = motion_diff(prev_crop, crop)

        recent_motion.append(motion)
        avg_motion = float(np.mean(recent_motion))
        prev_crop = crop.copy()

        sharp_ok = sharpness >= MIN_SHARPNESS
        bright_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        motion_ok = avg_motion <= MAX_MOTION_DIFF

        duplicate, current_hash = is_duplicate(crop, saved_hashes, MIN_HASH_DISTANCE)
        ready_to_capture = sharp_ok and bright_ok and motion_ok and (not duplicate)

        if ready_to_capture:
            status_text = "READY TO CAPTURE"
            status_color = (50, 180, 80)
        elif duplicate:
            status_text = "DUPLICATE FRAME"
            status_color = (0, 170, 255)
        elif not sharp_ok:
            status_text = "TOO BLURRY"
            status_color = (0, 0, 220)
        elif not bright_ok:
            status_text = "BAD LIGHTING"
            status_color = (0, 140, 255)
        else:
            status_text = "HOLD STILL"
            status_color = (0, 170, 255)

        now = time.time()
        if ready_to_capture and (now - last_capture_time >= MIN_CAPTURE_INTERVAL):
            save_path = person_dir / f"{PERSON_NAME}_{saved_count+1:04d}.jpg"
            save_crop(crop, save_path)

            if current_hash is None:
                current_hash = average_hash(cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE)))

            saved_hashes.append(current_hash)
            saved_count += 1
            last_capture_time = now

            print(
                f"Saved: {save_path.name} | "
                f"sharp={sharpness:.1f}, bright={brightness:.1f}, motion={avg_motion:.1f}"
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 170, 255), 2)

        cv2.putText(frame, "ALIGN EAR INSIDE BOX", (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        side_label = "RIGHT" if use_right_ear else "LEFT"
        draw_status_panel(
            frame, PERSON_NAME, saved_count, TARGET_IMAGES,
            status_text, status_color, sharpness, brightness,
            avg_motion, cam.mode, side_label
        )

        crop_preview = cv2.resize(crop, (260, 260))
        cv2.imshow("Ear Crop", crop_preview)
        cv2.imshow("Enrollment Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            use_right_ear = not use_right_ear
            prev_crop = None
            recent_motion.clear()
            print("Ear side toggled.")
        elif key == ord('s'):
            manual_hash = average_hash(cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE)))
            save_path = person_dir / f"{PERSON_NAME}_{saved_count+1:04d}.jpg"
            save_crop(crop, save_path)
            saved_hashes.append(manual_hash)
            saved_count += 1
            last_capture_time = time.time()
            print(f"Manual save: {save_path.name}")

        if saved_count >= TARGET_IMAGES:
            print(f"Target reached: {saved_count} images saved.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()