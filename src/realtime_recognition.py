import time
from pathlib import Path

import cv2
import numpy as np
import serial
import torch
from PIL import Image
from torchvision import transforms

from arcface_model import EarRecognitionModel

# =========================================================
# PATHS
# =========================================================
ROOT = Path(r"Z:\Code\EarEdge3")
MODEL_PATH = ROOT / "models" / "arcface_model.pth"
DB_PATH = ROOT / "models" / "embeddings.npy"

# =========================================================
# SETTINGS
# =========================================================
SIMILARITY_THRESHOLD = 0.68
MARGIN_THRESHOLD = 0.03

ACCESS_SIM_THRESHOLD = 0.75
ACCESS_MARGIN_THRESHOLD = 0.05

TOP_K = 3
REQUIRED_MATCHES = 7
UNLOCK_COOLDOWN = 5.0
RECOGNIZE_INTERVAL = 0.12
GRANT_HOLD_SECONDS = 2.5

AUTHORIZED_NAMES = None
# Example:
# AUTHORIZED_NAMES = {"Hari"}

# =========================================================
# DROIDCAM SETTINGS
# =========================================================
PHONE_IP = "172.30.10.243"
PORT = "4747"

# If DroidCam has username/password, fill these in.
DROIDCAM_USERNAME = ""
DROIDCAM_PASSWORD = ""

WINDOW_W = 1100
WINDOW_H = 720
USE_RIGHT_EAR = True

RIGHT_CROP = (0.58, 0.20, 0.92, 0.86)
LEFT_CROP = (0.08, 0.20, 0.42, 0.86)

# =========================================================
# ESP32 SETTINGS
# =========================================================
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================================================
# LOAD DATABASE
# =========================================================
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database file not found: {DB_PATH}")

database = np.load(DB_PATH, allow_pickle=True).item()

print("\nDatabase loaded:")
for name, db_embs in database.items():
    print(f"{name} -> {db_embs.shape}")

if len(database) == 0:
    raise RuntimeError("Embedding database is empty.")

# =========================================================
# LOAD MODEL
# =========================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = EarRecognitionModel(num_classes=264)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
state_dict.pop("arcface.weight", None)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# =========================================================
# HELPERS - SERIAL
# =========================================================
def connect_esp32():
    try:
        esp = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2.0)  # allow ESP32 reset after serial open
        esp.reset_input_buffer()
        esp.reset_output_buffer()
        print(f"Connected to ESP32 on {SERIAL_PORT}")

        try:
            esp.write(b"PING\n")
            time.sleep(0.3)
            while esp.in_waiting > 0:
                line = esp.readline().decode(errors="ignore").strip()
                if line:
                    print("ESP32 startup:", line)
        except Exception as e:
            print("ESP32 ping warning:", e)

        return esp

    except Exception as e:
        print("ESP32 not connected:", e)
        print("Recognition will still run, but unlock command will not be sent.")
        print("Troubleshooting tips:")
        print("1. Close Arduino IDE / Serial Monitor")
        print("2. Make sure COM port is correct")
        print("3. Unplug and replug the ESP32")
        print("4. Check whether another script is using the same COM port")
        return None


def send_unlock(esp32):
    if esp32 is not None and esp32.is_open:
        try:
            esp32.write(b"UNLOCK\n")
            esp32.flush()
            print("UNLOCK command sent.")
            return True
        except Exception as e:
            print("Failed to send UNLOCK:", e)
    else:
        print("ESP32 not available for unlock.")
    return False


def read_esp32_status(esp32, state):
    if esp32 is not None and esp32.is_open:
        try:
            while esp32.in_waiting > 0:
                line = esp32.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                print("ESP32:", line)

                if line == "READY":
                    state["lock_state"] = "READY"
                elif line == "UNLOCKING":
                    state["lock_state"] = "UNLOCKING"
                elif line == "OPEN":
                    state["lock_state"] = "OPEN"
                elif line == "LOCKING":
                    state["lock_state"] = "LOCKING"
                elif line == "LOCKED":
                    state["lock_state"] = "LOCKED"
                elif line == "BUSY":
                    state["lock_state"] = "BUSY"
                elif line == "CMD_OK":
                    state["lock_state"] = "CMD_OK"
        except Exception as e:
            print("ESP32 read error:", e)

# =========================================================
# HELPERS - CAMERA
# =========================================================
def build_droidcam_urls():
    urls = []

    if DROIDCAM_USERNAME and DROIDCAM_PASSWORD:
        urls.append(f"http://{DROIDCAM_USERNAME}:{DROIDCAM_PASSWORD}@{PHONE_IP}:{PORT}/video")

    urls.append(f"http://{PHONE_IP}:{PORT}/video")
    return urls


def open_camera():
    urls = build_droidcam_urls()

    for url in urls:
        print("Trying camera URL:", url.replace(DROIDCAM_PASSWORD, "****") if DROIDCAM_PASSWORD else url)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("Opened DroidCam stream.")
            return cap, url

        cap.release()

    return None, None

# =========================================================
# HELPERS - RECOGNITION
# =========================================================
def recognize(frame, top_k=TOP_K):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img)
        emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy()

    results = []

    for name, db_embs in database.items():
        scores = np.dot(emb, db_embs.T)[0]
        k = min(top_k, len(scores))
        top_scores = np.sort(scores)[-k:]
        score = float(np.mean(top_scores))
        results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = results[0]
    second_name, second_score = results[1] if len(results) > 1 else ("None", -1.0)
    margin = best_score - second_score

    if best_score < SIMILARITY_THRESHOLD:
        best_name = "Unknown"
        best_score = 0.0
        margin = 0.0

    print(
        f"Top1: {best_name} {best_score:.4f} | "
        f"Top2: {second_name} {second_score:.4f} | "
        f"Margin: {margin:.4f}"
    )

    return {
        "name": best_name,
        "score": best_score,
        "second_name": second_name,
        "second_score": second_score,
        "margin": margin,
    }


def compute_crop(frame, use_right_ear):
    h, w, _ = frame.shape
    rx1, ry1, rx2, ry2 = RIGHT_CROP if use_right_ear else LEFT_CROP
    x1, y1 = int(rx1 * w), int(ry1 * h)
    x2, y2 = int(rx2 * w), int(ry2 * h)
    return x1, y1, x2, y2


def smooth_box(prev_box, new_box, alpha=0.78):
    if prev_box is None:
        return new_box

    px1, py1, px2, py2 = prev_box
    nx1, ny1, nx2, ny2 = new_box

    sx1 = int(alpha * px1 + (1 - alpha) * nx1)
    sy1 = int(alpha * py1 + (1 - alpha) * ny1)
    sx2 = int(alpha * px2 + (1 - alpha) * nx2)
    sy2 = int(alpha * py2 + (1 - alpha) * ny2)

    return sx1, sy1, sx2, sy2


def is_authorized(name):
    if AUTHORIZED_NAMES is None:
        return True
    return name in AUTHORIZED_NAMES


def reset_state():
    return {
        "candidate_name": None,
        "candidate_count": 0,
        "last_name": "Unknown",
        "last_score": 0.0,
        "margin": 0.0,
        "status": "ACCESS DENIED",
        "message": "SHOW EAR TO CAMERA",
        "last_unlock_time": 0.0,
        "grant_hold_until": 0.0,
        "granted_name": "Unknown",
        "fps": 0.0,
        "fps_counter": 0,
        "fps_time": time.time(),
        "last_recog_time": 0.0,
        "reason": "waiting",
        "smoothed_box": None,
        "lock_state": "LOCKED",
    }

# =========================================================
# UI
# =========================================================
def draw_modern_ui(frame, state, side_label):
    h, w, _ = frame.shape
    overlay = frame.copy()

    cv2.rectangle(overlay, (15, 15), (540, 210), (18, 18, 24), -1)
    cv2.rectangle(overlay, (w - 290, 15), (w - 15, 210), (18, 18, 24), -1)
    cv2.rectangle(overlay, (15, h - 50), (700, h - 15), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    status = state["status"]
    if status == "ACCESS GRANTED":
        accent = (60, 200, 120)
    elif status == "AMBIGUOUS MATCH":
        accent = (0, 180, 255)
    else:
        accent = (80, 80, 230)

    cv2.rectangle(frame, (25, 25), (520, 68), accent, -1)
    cv2.rectangle(frame, (w - 280, 25), (w - 30, 33), accent, -1)

    cv2.putText(frame, status, (38, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    cv2.putText(frame, state["message"], (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (235, 235, 235), 2)

    cv2.putText(frame, f"Top Match: {state['last_name']} ({state['last_score']:.2f})", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.67, (180, 255, 180), 2)

    cv2.putText(frame, f"Margin: {state['margin']:.2f}  Stable: {state['candidate_count']}/{REQUIRED_MATCHES}", (30, 163),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)

    cv2.putText(frame, "SYSTEM", (w - 265, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {state['fps']:.1f}", (w - 265, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, f"Ear: {side_label}", (w - 265, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, "Cam: DroidCam", (w - 265, 152),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, f"Lock: {state['lock_state']}", (w - 265, 178),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    footer = "q quit   u unlock   e toggle ear   r reset"
    cv2.putText(frame, footer, (28, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    esp32 = connect_esp32()
    cap, active_url = open_camera()

    if cap is None:
        print("Cannot open DroidCam stream.")
        print("Check:")
        print("1. Phone and PC are on same network")
        print("2. IP address is correct")
        print("3. DroidCam is running on phone")
        print("4. Port is correct")
        print("5. Username/password are correct if enabled")
        if esp32 is not None and esp32.is_open:
            esp32.close()
        raise SystemExit(1)

    print(f"Using camera stream: {active_url}")
    print("Press 'q' to exit")
    print("Press 'u' for manual unlock test")
    print("Press 'e' to toggle ear side")
    print("Press 'r' to reset tracking")

    state = reset_state()
    frame_count = 0
    no_frame_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not ret or frame is None:
            no_frame_count += 1

            if frame_count % 100 == 0:
                print(f"No frame received (attempt {frame_count})")

            if no_frame_count >= 30:
                print("Reconnecting to DroidCam stream...")
                cap.release()
                time.sleep(1.0)
                cap, active_url = open_camera()
                no_frame_count = 0

                if cap is None:
                    print("Reconnection failed. Retrying...")
                    time.sleep(1.0)
                    continue

            time.sleep(0.01)
            continue

        no_frame_count = 0
        frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

        read_esp32_status(esp32, state)

        state["fps_counter"] += 1
        now = time.time()
        elapsed = now - state["fps_time"]

        if elapsed >= 1.0:
            state["fps"] = state["fps_counter"] / elapsed
            state["fps_counter"] = 0
            state["fps_time"] = now

        is_holding_grant = now < state["grant_hold_until"]

        raw_box = compute_crop(frame, USE_RIGHT_EAR)
        state["smoothed_box"] = smooth_box(state["smoothed_box"], raw_box, alpha=0.78)
        x1, y1, x2, y2 = state["smoothed_box"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 170, 255), 2)

        if (not is_holding_grant) and (now - state["last_recog_time"] >= RECOGNIZE_INTERVAL):
            state["last_recog_time"] = now

            ear_crop = frame[y1:y2, x1:x2]
            if ear_crop.size != 0:
                model_crop = cv2.resize(ear_crop, (224, 224))
                result = recognize(model_crop)

                name = result["name"]
                score = result["score"]
                margin = result["margin"]

                state["last_name"] = name
                state["last_score"] = score
                state["margin"] = margin

                display_valid = score >= SIMILARITY_THRESHOLD and margin >= MARGIN_THRESHOLD
                access_valid = score >= ACCESS_SIM_THRESHOLD and margin >= ACCESS_MARGIN_THRESHOLD

                if not display_valid:
                    state["candidate_name"] = None
                    state["candidate_count"] = 0
                    state["last_name"] = "Unknown"

                    if margin < MARGIN_THRESHOLD:
                        state["status"] = "AMBIGUOUS MATCH"
                        state["message"] = "PLEASE HOLD STILL"
                    else:
                        state["status"] = "ACCESS DENIED"
                        state["message"] = "SHOW EAR TO CAMERA"

                else:
                    if state["candidate_name"] == name:
                        state["candidate_count"] += 1
                    else:
                        state["candidate_name"] = name
                        state["candidate_count"] = 1

                    if state["candidate_count"] >= REQUIRED_MATCHES and access_valid:
                        state["last_name"] = name
                        state["last_score"] = score

                        if is_authorized(name):
                            state["status"] = "ACCESS GRANTED"
                            state["message"] = f"WELCOME {name.upper()}"
                            state["granted_name"] = name
                            state["grant_hold_until"] = now + GRANT_HOLD_SECONDS

                            if now - state["last_unlock_time"] > UNLOCK_COOLDOWN:
                                if send_unlock(esp32):
                                    state["last_unlock_time"] = now
                        else:
                            state["status"] = "ACCESS DENIED"
                            state["message"] = f"{name.upper()} NOT AUTHORIZED"
                    else:
                        state["status"] = "ACCESS DENIED"
                        state["message"] = f"VERIFYING {name.upper()}"

                cv2.imshow("Ear Crop", model_crop)

        if now < state["grant_hold_until"]:
            state["status"] = "ACCESS GRANTED"
            state["message"] = f"WELCOME {state['granted_name'].upper()}"
            state["last_name"] = state["granted_name"]

        side_label = "RIGHT" if USE_RIGHT_EAR else "LEFT"
        draw_modern_ui(frame, state, side_label)

        cv2.imshow("Ear Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            print("Manual unlock triggered.")
            send_unlock(esp32)
        elif key == ord('e'):
            USE_RIGHT_EAR = not USE_RIGHT_EAR
            state = reset_state()
            print("Ear side toggled.")
        elif key == ord('r'):
            state = reset_state()
            print("Tracking reset.")

    cap.release()

    if esp32 is not None and esp32.is_open:
        esp32.close()

    cv2.destroyAllWindows()