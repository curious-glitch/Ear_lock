import threading
import time
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

import realtime_recognition as core


# =========================
# APP SETTINGS
# =========================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_W = 1350
APP_H = 820
PREVIEW_W = 900
PREVIEW_H = 590


# =========================
# MAIN APP
# =========================
app = ctk.CTk()
app.geometry(f"{APP_W}x{APP_H}")
app.title("Ear Recognition Access System")
app.resizable(False, False)

running = False
camera_thread = None
state = None
cap = None
log_cache = []


# =========================
# HELPERS
# =========================
def set_status_visual(status_text: str):
    if status_text == "ACCESS GRANTED":
        status_label.configure(text=status_text, text_color="#4CAF50")
    elif status_text == "ACCESS DENIED":
        status_label.configure(text=status_text, text_color="#F44336")
    elif status_text == "AMBIGUOUS MATCH":
        status_label.configure(text=status_text, text_color="#FFC107")
    else:
        status_label.configure(text=status_text, text_color="#E0E0E0")


def safe_text(value):
    return str(value) if value is not None else "-"


def add_log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    if line in log_cache[-3:]:
        return
    log_cache.append(line)
    log_box.insert("end", line + "\n")
    log_box.see("end")


def update_connection_labels():
    if core.esp32 is not None and core.esp32.is_open:
        esp32_label.configure(text="ESP32: Connected", text_color="#4CAF50")
    else:
        esp32_label.configure(text="ESP32: Not Connected", text_color="#F44336")

    camera_label.configure(text="Camera: Running" if running else "Camera: Stopped",
                           text_color="#4CAF50" if running else "#F44336")


def update_info_panel(local_state):
    set_status_visual(local_state["status"])
    message_label.configure(text=f"Message: {safe_text(local_state['message'])}")
    match_label.configure(
        text=f"Top Match: {safe_text(local_state['last_name'])} ({local_state['last_score']:.2f})"
    )
    margin_label.configure(text=f"Margin: {local_state['margin']:.2f}")
    stable_label.configure(
        text=f"Stable Matches: {local_state['candidate_count']}/{core.REQUIRED_MATCHES}"
    )
    lock_label.configure(text=f"Lock: {safe_text(local_state['lock_state'])}")
    fps_label.configure(text=f"FPS: {local_state['fps']:.1f}")
    ear_label.configure(text=f"Ear Side: {'RIGHT' if core.USE_RIGHT_EAR else 'LEFT'}")


def reset_ui_only():
    set_status_visual("Status: Idle")
    message_label.configure(text="Message: Show ear to camera")
    match_label.configure(text="Top Match: -")
    margin_label.configure(text="Margin: -")
    stable_label.configure(text=f"Stable Matches: 0/{core.REQUIRED_MATCHES}")
    lock_label.configure(text="Lock: LOCKED")
    fps_label.configure(text="FPS: 0.0")
    ear_label.configure(text=f"Ear Side: {'RIGHT' if core.USE_RIGHT_EAR else 'LEFT'}")


# =========================
# CAMERA LOOP
# =========================
def camera_loop():
    global running, state, cap

    try:
        cap = core.open_camera()
        if not cap.isOpened():
            add_log("Failed to open camera stream.")
            running = False
            app.after(0, update_connection_labels)
            return

        state = core.reset_state()
        frame_count = 0
        no_frame_count = 0
        last_log_status = None

        add_log("Camera started.")

        while running:
            ret, frame = cap.read()
            frame_count += 1

            if not ret or frame is None:
                no_frame_count += 1

                if no_frame_count >= 30:
                    add_log("Reconnecting camera stream...")
                    cap.release()
                    time.sleep(1.0)
                    cap = core.open_camera()
                    no_frame_count = 0

                time.sleep(0.01)
                continue

            no_frame_count = 0
            frame = cv2.resize(frame, (core.WINDOW_W, core.WINDOW_H))

            # Read ESP32 status
            core.read_esp32_status(state)

            # FPS
            state["fps_counter"] += 1
            now = time.time()
            elapsed = now - state["fps_time"]
            if elapsed >= 1.0:
                state["fps"] = state["fps_counter"] / elapsed
                state["fps_counter"] = 0
                state["fps_time"] = now

            is_holding_grant = now < state["grant_hold_until"]

            # Crop box
            raw_box = core.compute_crop(frame, core.USE_RIGHT_EAR)
            state["smoothed_box"] = core.smooth_box(state["smoothed_box"], raw_box, alpha=0.78)
            x1, y1, x2, y2 = state["smoothed_box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 170, 255), 2)

            # Recognition on timer
            if (not is_holding_grant) and (now - state["last_recog_time"] >= core.RECOGNIZE_INTERVAL):
                state["last_recog_time"] = now

                ear_crop = frame[y1:y2, x1:x2]
                if ear_crop.size != 0:
                    model_crop = cv2.resize(ear_crop, (224, 224))
                    result = core.recognize(model_crop)

                    name = result["name"]
                    score = result["score"]
                    margin = result["margin"]

                    state["last_name"] = name
                    state["last_score"] = score
                    state["margin"] = margin

                    display_valid = score >= core.SIMILARITY_THRESHOLD and margin >= core.MARGIN_THRESHOLD
                    access_valid = score >= core.ACCESS_SIM_THRESHOLD and margin >= core.ACCESS_MARGIN_THRESHOLD

                    if not display_valid:
                        state["candidate_name"] = None
                        state["candidate_count"] = 0
                        state["last_name"] = "Unknown"

                        if margin < core.MARGIN_THRESHOLD:
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

                        if state["candidate_count"] >= core.REQUIRED_MATCHES and access_valid:
                            state["last_name"] = name
                            state["last_score"] = score

                            if core.is_authorized(name):
                                state["status"] = "ACCESS GRANTED"
                                state["message"] = f"WELCOME {name.upper()}"
                                state["granted_name"] = name
                                state["grant_hold_until"] = now + core.GRANT_HOLD_SECONDS

                                if now - state["last_unlock_time"] > core.UNLOCK_COOLDOWN:
                                    if core.send_unlock():
                                        state["last_unlock_time"] = now
                            else:
                                state["status"] = "ACCESS DENIED"
                                state["message"] = f"{name.upper()} NOT AUTHORIZED"
                        else:
                            state["status"] = "VERIFYING"
                            state["message"] = f"VERIFYING {name.upper()}"

            if now < state["grant_hold_until"]:
                state["status"] = "ACCESS GRANTED"
                state["message"] = f"WELCOME {state['granted_name'].upper()}"
                state["last_name"] = state["granted_name"]

            # Overlay title
            cv2.rectangle(frame, (15, 15), (460, 60), (18, 18, 24), -1)
            cv2.putText(frame, "EAR-BASED SMART ACCESS SYSTEM", (28, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

            # Convert for Tkinter preview
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            preview_img = Image.fromarray(preview)
            preview_tk = ImageTk.PhotoImage(preview_img)

            def refresh_ui():
                video_label.configure(image=preview_tk, text="")
                video_label.image = preview_tk
                update_info_panel(state)
                update_connection_labels()

            app.after(0, refresh_ui)

            current_log_status = f"{state['status']} | {state['last_name']} | {state['lock_state']}"
            if current_log_status != last_log_status:
                add_log(current_log_status)
                last_log_status = current_log_status

            time.sleep(0.01)

    except Exception as e:
        add_log(f"App error: {e}")
    finally:
        running = False
        if cap is not None:
            cap.release()
        app.after(0, update_connection_labels)
        add_log("Camera stopped.")


# =========================
# BUTTON ACTIONS
# =========================
def start_camera():
    global running, camera_thread
    if running:
        add_log("Camera is already running.")
        return

    running = True
    update_connection_labels()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()


def stop_camera():
    global running
    running = False
    update_connection_labels()


def manual_unlock():
    if core.send_unlock():
        add_log("Manual unlock command sent.")
    else:
        add_log("Manual unlock failed.")


def toggle_ear():
    core.USE_RIGHT_EAR = not core.USE_RIGHT_EAR
    ear_label.configure(text=f"Ear Side: {'RIGHT' if core.USE_RIGHT_EAR else 'LEFT'}")
    add_log(f"Ear side changed to {'RIGHT' if core.USE_RIGHT_EAR else 'LEFT'}.")


def reset_system():
    global state
    if state is not None:
        state = core.reset_state()
    reset_ui_only()
    add_log("System reset.")


def clear_logs():
    log_box.delete("1.0", "end")
    log_cache.clear()


def on_close():
    global running
    running = False
    time.sleep(0.2)
    app.destroy()


# =========================
# LAYOUT
# =========================
title_label = ctk.CTkLabel(
    app,
    text="EAR RECOGNITION ACCESS SYSTEM",
    font=("Arial", 28, "bold")
)
title_label.pack(pady=(12, 6))

main_frame = ctk.CTkFrame(app, corner_radius=16)
main_frame.pack(fill="both", expand=True, padx=16, pady=10)

left_frame = ctk.CTkFrame(main_frame, corner_radius=16)
left_frame.pack(side="left", fill="both", expand=False, padx=12, pady=12)

right_frame = ctk.CTkFrame(main_frame, corner_radius=16, width=360)
right_frame.pack(side="right", fill="y", padx=12, pady=12)
right_frame.pack_propagate(False)

video_label = ctk.CTkLabel(left_frame, text="Camera preview will appear here",
                           width=PREVIEW_W, height=PREVIEW_H)
video_label.pack(padx=10, pady=10)

button_frame = ctk.CTkFrame(left_frame, corner_radius=12)
button_frame.pack(fill="x", padx=10, pady=(0, 10))

start_btn = ctk.CTkButton(button_frame, text="Start Camera", command=start_camera, width=130)
start_btn.grid(row=0, column=0, padx=8, pady=10)

stop_btn = ctk.CTkButton(button_frame, text="Stop Camera", command=stop_camera, width=130)
stop_btn.grid(row=0, column=1, padx=8, pady=10)

unlock_btn = ctk.CTkButton(button_frame, text="Manual Unlock", command=manual_unlock, width=130)
unlock_btn.grid(row=0, column=2, padx=8, pady=10)

toggle_btn = ctk.CTkButton(button_frame, text="Toggle Ear", command=toggle_ear, width=130)
toggle_btn.grid(row=0, column=3, padx=8, pady=10)

reset_btn = ctk.CTkButton(button_frame, text="Reset", command=reset_system, width=130)
reset_btn.grid(row=0, column=4, padx=8, pady=10)

exit_btn = ctk.CTkButton(button_frame, text="Exit", command=on_close, width=130, fg_color="#B71C1C", hover_color="#8E0000")
exit_btn.grid(row=0, column=5, padx=8, pady=10)

# Right-side information panel
section_title = ctk.CTkLabel(right_frame, text="System Monitor", font=("Arial", 22, "bold"))
section_title.pack(pady=(14, 10))

status_label = ctk.CTkLabel(right_frame, text="Status: Idle", font=("Arial", 22, "bold"))
status_label.pack(pady=8)

message_label = ctk.CTkLabel(right_frame, text="Message: Show ear to camera", font=("Arial", 15), wraplength=320)
message_label.pack(pady=6)

match_label = ctk.CTkLabel(right_frame, text="Top Match: -", font=("Arial", 15))
match_label.pack(pady=6)

margin_label = ctk.CTkLabel(right_frame, text="Margin: -", font=("Arial", 15))
margin_label.pack(pady=6)

stable_label = ctk.CTkLabel(right_frame, text=f"Stable Matches: 0/{core.REQUIRED_MATCHES}", font=("Arial", 15))
stable_label.pack(pady=6)

lock_label = ctk.CTkLabel(right_frame, text="Lock: LOCKED", font=("Arial", 15))
lock_label.pack(pady=6)

fps_label = ctk.CTkLabel(right_frame, text="FPS: 0.0", font=("Arial", 15))
fps_label.pack(pady=6)

ear_label = ctk.CTkLabel(right_frame, text=f"Ear Side: {'RIGHT' if core.USE_RIGHT_EAR else 'LEFT'}", font=("Arial", 15))
ear_label.pack(pady=6)

camera_label = ctk.CTkLabel(right_frame, text="Camera: Stopped", font=("Arial", 15))
camera_label.pack(pady=6)

esp32_label = ctk.CTkLabel(right_frame, text="ESP32: Checking...", font=("Arial", 15))
esp32_label.pack(pady=6)

log_title = ctk.CTkLabel(right_frame, text="Recognition Log", font=("Arial", 18, "bold"))
log_title.pack(pady=(18, 8))

log_box = ctk.CTkTextbox(right_frame, width=320, height=210)
log_box.pack(padx=12, pady=6)

clear_log_btn = ctk.CTkButton(right_frame, text="Clear Logs", command=clear_logs, width=150)
clear_log_btn.pack(pady=10)

reset_ui_only()
update_connection_labels()
add_log("App loaded.")

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()