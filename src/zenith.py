import ctypes
import json
import os
import time
import cv2
import keyboard as keyboard
import numpy as np
import dxcam
import requests
import win32api
import win32con
import tkinter as tk
from tkinter import ttk
from tkinter import Entry
from ttkthemes import ThemedStyle

from yolov8 import YOLOv8

# ------- CONFIG --------

# ----- Aim Config -----
MOVEMENT_AMP_X = 1
MOVEMENT_AMP_Y = 1
MOUSE_MOVEMENT_THRESHOLD = 1
TOGGLE_AIM_KEY = 'x'
TB_HEAD_OFFSET = 50

# --- Triggerbot Config ---
TB_THRESHOLD = 0.2
ENABLE_TB = True
TOGGLE_TB_KEY = 'z'

# ----- Settings Confing -----
WINDOW_WIDTH, WINDOW_HEIGHT = 480, 480

# ----- Debug Config -----

# Text
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 2

# Boxes
RECTANGLE_COLOR = (0, 36, 255)
RECTANGLE_THICKNESS = 2
FILLED_RECTANGLE = True
RECTANGLE_OPACITY = 0.3

# Lines
LINE_THICKNESS = 2
LINE_COLOR = (0, 255, 0)
LINE_OPACITY = 0.5

# Triggerbot
TB_INDICATOR_COLOR = (0, 255, 255)

# Misc
DISPLAY_FPS = True
DISPLAY_CLOSEST = True
DISPLAY_ENABLED = True

# ------ END CONFIG --------

# Ignore in the GUI
SCREEN_WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)
MODEL_PATH = "models/humans.onnx"
CONF_THRES = 0.5
IOU_THRES = 0.5
DETECTION_CLASS_ID = 0


def save_config():
    config_data = {
        "Setup": {
            "WINDOW_WIDTH": WINDOW_WIDTH,
            "WINDOW_HEIGHT": WINDOW_HEIGHT,
            "MODEL_PATH": MODEL_PATH
        },
        "Aim": {
            "GAME_X": MOVEMENT_AMP_X,
            "GAME_Y": MOVEMENT_AMP_Y,
            "MOUSE_MOVEMENT_THRESHOLD": MOUSE_MOVEMENT_THRESHOLD,
            "TOGGLE_AIM_KEY": TOGGLE_AIM_KEY
        },
        "Triggerbot": {
            "TB_THRESHOLD": TB_THRESHOLD,
            "ENABLE_TB": ENABLE_TB,
            "TOGGLE_TB_KEY": TOGGLE_TB_KEY,
            "TB_HEAD_OFFSET": TB_HEAD_OFFSET  # Add this line
        },
        "Debug": {
            # ... (Add other debug parameters)
        }
    }

    with open("config.json", "w") as config_file:
        json.dump(config_data, config_file)


def load_gui_config():
    config_gui.mov_amp_x_entry.delete(0, tk.END)
    config_gui.mov_amp_x_entry.insert(0, str(MOVEMENT_AMP_X))

    config_gui.mov_amp_y_entry.delete(0, tk.END)
    config_gui.mov_amp_y_entry.insert(0, str(MOVEMENT_AMP_Y))

    config_gui.aim_thres.delete(0, tk.END)
    config_gui.aim_thres.insert(0, str(MOUSE_MOVEMENT_THRESHOLD))

    config_gui.toggle_aim_key_entry.delete(0, tk.END)
    config_gui.toggle_aim_key_entry.insert(0, TOGGLE_AIM_KEY)

    config_gui.tb_threshold_entry.delete(0, tk.END)
    config_gui.tb_threshold_entry.insert(0, str(TB_THRESHOLD))

    config_gui.toggle_tb_key_entry.delete(0, tk.END)
    config_gui.toggle_tb_key_entry.insert(0, TOGGLE_TB_KEY)

    config_gui.tb_head_offset_entry.delete(0, tk.END)
    config_gui.tb_head_offset_entry.insert(0, str(TB_HEAD_OFFSET))

    config_gui.window_width_entry.delete(0, tk.END)
    config_gui.window_width_entry.insert(0, str(WINDOW_WIDTH))

    config_gui.window_height_entry.delete(0, tk.END)
    config_gui.window_height_entry.insert(0, str(WINDOW_HEIGHT))

    config_gui.model_path_entry.delete(0, tk.END)
    config_gui.model_path_entry.insert(0, MODEL_PATH)


def load_config():
    try:
        with open("config.json", "r") as config_file:
            # Check if the file is empty
            if not config_file.read(1):
                raise ValueError("Empty file")

            config_file.seek(0)  # Reset file pointer to the beginning
            config_data = json.load(config_file)

            # Update variables with values from the loaded config
            global MOUSE_MOVEMENT_THRESHOLD, TOGGLE_AIM_KEY, MOVEMENT_AMP_X, MOVEMENT_AMP_Y
            global TB_THRESHOLD, TOGGLE_TB_KEY, TB_HEAD_OFFSET
            global WINDOW_WIDTH, WINDOW_HEIGHT, MODEL_PATH
            # ... (Update other parameters)

            setup_config = config_data.get("Setup", {})
            WINDOW_WIDTH = setup_config.get("WINDOW_WIDTH", WINDOW_WIDTH)
            WINDOW_HEIGHT = setup_config.get("WINDOW_HEIGHT", WINDOW_HEIGHT)
            MODEL_PATH = setup_config.get("MODEL_PATH", MODEL_PATH)

            MOVEMENT_AMP_X = config_data.get("Aim", {}).get("AMP_X", MOVEMENT_AMP_X)
            MOVEMENT_AMP_Y = config_data.get("Aim", {}).get("AMP_Y", MOVEMENT_AMP_Y)
            MOUSE_MOVEMENT_THRESHOLD = config_data.get("Aim", {}).get("MOUSE_MOVEMENT_THRESHOLD",
                                                                      MOUSE_MOVEMENT_THRESHOLD)
            TOGGLE_AIM_KEY = config_data.get("Aim", {}).get("TOGGLE_AIM_KEY", TOGGLE_AIM_KEY)
            # ... (Update other parameters)

            TB_THRESHOLD = config_data.get("Triggerbot", {}).get("TB_THRESHOLD", TB_THRESHOLD)
            TOGGLE_TB_KEY = config_data.get("Triggerbot", {}).get("TOGGLE_TB_KEY", TOGGLE_TB_KEY)
            TB_HEAD_OFFSET = config_data.get("Triggerbot", {}).get("TB_HEAD_OFFSET", TOGGLE_TB_KEY)
            # ... (Update other parameters)

            # Update the GUI with the loaded configuration
            load_gui_config()

    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        load_gui_config()
        pass


class ConfigGUI:
    def __init__(self, master):
        self.master = master
        master.title("Zenith")

        # Apply ThemedStyle for a modern look
        style = ThemedStyle(master)
        style.set_theme("plastik")  # You can choose a different theme

        # Setup Config Section
        setup_frame = ttk.LabelFrame(master, text="Setup Config", padding=(20, 10))
        setup_frame.grid(row=2, column=0, padx=20, pady=20, sticky="w")

        ttk.Label(setup_frame, text="Window Width:").grid(row=0, column=0, sticky="e", pady=5)
        self.window_width_entry = Entry(setup_frame)
        self.window_width_entry.grid(row=0, column=1, pady=5)

        ttk.Label(setup_frame, text="Window Height:").grid(row=1, column=0, sticky="e", pady=5)
        self.window_height_entry = Entry(setup_frame)
        self.window_height_entry.grid(row=1, column=1, pady=5)

        ttk.Label(setup_frame, text="Model Path:").grid(row=2, column=0, sticky="e", pady=5)
        self.model_path_entry = Entry(setup_frame)
        self.model_path_entry.grid(row=2, column=1, pady=5)

        # Aim Config Section
        aim_frame = ttk.LabelFrame(master, text="Aim Config", padding=(20, 10))
        aim_frame.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        ttk.Label(aim_frame, text="Movement Amp X:").grid(row=0, column=0, sticky="e", pady=5)
        self.mov_amp_x_entry = Entry(aim_frame)
        self.mov_amp_x_entry.grid(row=0, column=1, pady=5)

        ttk.Label(aim_frame, text="Movement Amp Y:").grid(row=1, column=0, sticky="e", pady=5)
        self.mov_amp_y_entry = Entry(aim_frame)
        self.mov_amp_y_entry.grid(row=1, column=1, pady=5)

        ttk.Label(aim_frame, text="Aim Threshold:").grid(row=2, column=0, sticky="e", pady=5)
        self.aim_thres = Entry(aim_frame)
        self.aim_thres.grid(row=2, column=1, pady=5)

        ttk.Label(aim_frame, text="Toggle Aim Key:").grid(row=3, column=0, sticky="e", pady=5)
        self.toggle_aim_key_entry = Entry(aim_frame)
        self.toggle_aim_key_entry.grid(row=3, column=1, pady=5)

        # Triggerbot Config Section
        tb_frame = ttk.LabelFrame(master, text="Triggerbot Config", padding=(20, 10))
        tb_frame.grid(row=1, column=0, padx=20, pady=20, sticky="w")

        ttk.Label(tb_frame, text="Triggerbot Threshold:").grid(row=0, column=0, sticky="e", pady=5)
        self.tb_threshold_entry = Entry(tb_frame)
        self.tb_threshold_entry.grid(row=0, column=1, pady=5)

        # Add this entry for TB_HEAD_OFFSET
        ttk.Label(tb_frame, text="Triggerbot Head Offset:").grid(row=2, column=0, sticky="e", pady=5)
        self.tb_head_offset_entry = Entry(tb_frame)
        self.tb_head_offset_entry.grid(row=2, column=1, pady=5)

        ttk.Label(tb_frame, text="Toggle Triggerbot Key:").grid(row=1, column=0, sticky="e", pady=5)
        self.toggle_tb_key_entry = Entry(tb_frame)
        self.toggle_tb_key_entry.grid(row=1, column=1, pady=5)

        # Apply Button
        apply_button = ttk.Button(master, text="Apply Config", command=self.apply_config)
        apply_button.grid(row=4, column=0, pady=20)

    def apply_config(self):
        global MOVEMENT_AMP_X, MOVEMENT_AMP_Y, MOUSE_MOVEMENT_THRESHOLD, TB_THRESHOLD, tb_enabled, yolov8_detector
        global TOGGLE_AIM_KEY, TB_HEAD_OFFSET, TOGGLE_TB_KEY, WINDOW_WIDTH, WINDOW_HEIGHT, MODEL_PATH

        # Retrieve values from GUI and apply them to the corresponding variables in your code
        WINDOW_WIDTH = int(self.window_width_entry.get())
        WINDOW_HEIGHT = int(self.window_height_entry.get())
        MODEL_PATH = self.model_path_entry.get()

        MOVEMENT_AMP_X = float(self.mov_amp_x_entry.get())
        MOVEMENT_AMP_Y = float(self.mov_amp_y_entry.get())
        MOUSE_MOVEMENT_THRESHOLD = float(self.aim_thres.get())
        TOGGLE_AIM_KEY = self.toggle_aim_key_entry.get()

        TB_THRESHOLD = float(self.tb_threshold_entry.get())
        TOGGLE_TB_KEY = self.toggle_tb_key_entry.get()
        TB_HEAD_OFFSET = float(self.tb_head_offset_entry.get())

        # ... (apply values for other triggerbot config parameters)
        # Save the configuration to the JSON file
        save_config()
        yolov8_detector = YOLOv8(MODEL_PATH, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
        cv2.destroyAllWindows()
        camera.stop()
        camera.start(region=((SCREEN_WIDTH - WINDOW_WIDTH) // 2, (SCREEN_HEIGHT - WINDOW_HEIGHT) // 2,
                             (SCREEN_WIDTH + WINDOW_WIDTH) // 2, (SCREEN_HEIGHT + WINDOW_HEIGHT) // 2))
        time.sleep(0.2)


# Create the main window and the configuration GUI
root = tk.Tk()
config_gui = ConfigGUI(root)

# Load the configuration when the program starts
load_config()


def update_gui():
    root.update_idletasks()
    root.update()


# Initialize camera and frame capture
camera = dxcam.create()
camera.start(region=((SCREEN_WIDTH - WINDOW_WIDTH) // 2, (SCREEN_HEIGHT - WINDOW_HEIGHT) // 2,
                     (SCREEN_WIDTH + WINDOW_WIDTH) // 2, (SCREEN_HEIGHT + WINDOW_HEIGHT) // 2))

MODEL_FILENAMES = ["apex.onnx", "cs2.onnx", "fortnite.onnx", "humans.onnx", "phantomforces.onnx", "roblox.onnx"]
MODEL_URLS = ["https://github.com/moist-socks/zenith/raw/main/models/apex.onnx",
              "https://github.com/moist-socks/zenith/raw/main/models/cs2.onnx",
              "https://github.com/moist-socks/zenith/raw/main/models/fortnite.onnx",
              "https://github.com/moist-socks/zenith/raw/main/models/humans.onnx",
              "https://github.com/moist-socks/zenith/raw/main/models/phantomforces.onnx",
              "https://github.com/moist-socks/zenith/raw/main/models/roblox.onnx"]

# Initialize YOLOv8 model with adjustable parameters
# Check if the model directory exists
if not os.path.exists("models"):
    # If not, create the directory
    os.makedirs("models")

# Iterate through the models
for filename, url in zip(MODEL_FILENAMES, MODEL_URLS):
    model_path = os.path.join("models", filename)

    # Check if the model file exists
    if not os.path.exists(model_path):
        # If not, download the file
        response = requests.get(url)

        if response.status_code == 200:
            with open(model_path, 'wb') as model_file:
                model_file.write(response.content)

            print(f"Model '{filename}' downloaded and saved to: {model_path}")
        else:
            print(f"Failed to download the model '{filename}'. HTTP status code: {response.status_code}")
            exit()
    else:
        print(f"Model '{filename}' already exists at: {model_path}")

yolov8_detector = YOLOv8(MODEL_PATH, conf_thres=CONF_THRES, iou_thres=IOU_THRES)

# FPS tracking
prev_time = time.time()
highest_fps = 0.0

# Variables for mouse click state
left_click_pressed = False
tb_enabled = True
mouse_movement_enabled = True

old_rectangle_color = RECTANGLE_COLOR

# Main loop
try:
    while True:
        # Capture screenshot
        screenshot = camera.get_latest_frame()
        frame = np.array(screenshot)

        # Perform object detection
        boxes, scores, class_ids = yolov8_detector(frame)

        # Calculate center of the window
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

        # Process detected objects based on user-defined class ID
        combined_img = frame.copy()
        closest_detection = None
        closest_distance = float('inf')

        # Draw filled or outlined rectangle based on customization with opacity control
        for x_min, y_min, x_max, y_max in ((map(int, box)) for box, class_id in zip(boxes, class_ids) if
                                           class_id == DETECTION_CLASS_ID):
            if FILLED_RECTANGLE:
                rectangle_img = np.zeros_like(frame)
                cv2.rectangle(rectangle_img, (x_min, y_min), (x_max, y_max), RECTANGLE_COLOR, thickness=cv2.FILLED)
                combined_img = cv2.addWeighted(combined_img, 1.0, rectangle_img, RECTANGLE_OPACITY, 0)
            else:
                cv2.rectangle(combined_img, (x_min, y_min), (x_max, y_max), RECTANGLE_COLOR,
                              thickness=RECTANGLE_THICKNESS)

            # Calculate the center of the detection
            detection_center_x, detection_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

            if config_gui.tb_head_offset_entry.get():
                # Calculate the offset based on a percentage of the box height
                calculated_head_offset = float(config_gui.tb_head_offset_entry.get())
                box_height = y_max - y_min
                offset_percent = calculated_head_offset / 100
                calculated_head_offset = int(box_height * offset_percent)

                # Adjust the y coordinate based on TB_HEAD_OFFSET (subtracting for moving upwards)
                detection_center_y -= calculated_head_offset

            # Calculate distance from window center to detection center
            distance = np.linalg.norm(
                np.array([center_x, center_y]) - np.array([detection_center_x, detection_center_y]))

            # Update closest detection
            if distance < closest_distance:
                closest_detection = (detection_center_x, detection_center_y)
                closest_distance = distance

        # Toggle Triggerbot on/off with 't' key
        if keyboard.is_pressed(TOGGLE_TB_KEY):
            tb_enabled = not tb_enabled
            print(f'Toggle Triggerbot: {tb_enabled}')  # Console log
            time.sleep(0.2)  # Delay to prevent multiple toggles on a single key press

        # Toggle Mouse Movement on/off with 'm' key
        if keyboard.is_pressed(TOGGLE_AIM_KEY):
            mouse_movement_enabled = not mouse_movement_enabled
            print(f'Toggle Mouse Movement: {mouse_movement_enabled}')  # Console log
            time.sleep(0.2)  # Delay to prevent multiple toggles on a single key press

        # Simulate left-click if Triggerbot is enabled and the center enters a detected box
        if tb_enabled and closest_distance < TB_THRESHOLD * min(WINDOW_WIDTH, WINDOW_HEIGHT):
            if not left_click_pressed:
                ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                left_click_pressed = True
        elif left_click_pressed:
            ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            left_click_pressed = False

        # Change rectangle color based on TB_INDICATOR_COLOR
        RECTANGLE_COLOR = TB_INDICATOR_COLOR if TB_INDICATOR_COLOR and closest_distance < TB_THRESHOLD * min(
            WINDOW_WIDTH, WINDOW_HEIGHT) else old_rectangle_color

        # Move the mouse cursor to the center of the nearest bounding box if Caps Lock is pressed and Mouse Movement
        # is enabled
        if closest_detection is not None and mouse_movement_enabled:
            # Calculate displacement
            displacement_x = (closest_detection[0] - center_x) * MOVEMENT_AMP_X
            displacement_y = (closest_detection[1] - center_y) * MOVEMENT_AMP_Y

            # Check if the displacement exceeds the threshold before triggering mouse movement
            if abs(displacement_x) > MOUSE_MOVEMENT_THRESHOLD or abs(displacement_y) > MOUSE_MOVEMENT_THRESHOLD:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(displacement_x * MOVEMENT_AMP_X),
                                     int(displacement_y * MOVEMENT_AMP_Y), 0, 0)

        # Draw a line from the center to the closest detection if enabled
        if DISPLAY_CLOSEST and closest_detection is not None:
            closest_detection = tuple(map(int, closest_detection))  # Convert to integers
            line_color_with_opacity = (*LINE_COLOR, int(255 * LINE_OPACITY))  # Adjust opacity
            cv2.line(combined_img, (center_x, center_y), closest_detection, line_color_with_opacity,
                     thickness=LINE_THICKNESS)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        highest_fps = max(highest_fps, fps)

        # Display FPS on the frame if enabled
        if DISPLAY_FPS:
            fps_text = f"FPS: {fps:.2f} | Highest: {highest_fps:.2f}"
            cv2.putText(combined_img, fps_text, (10, 30), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        # Show the frame if enabled
        if DISPLAY_ENABLED:
            cv2.imshow("Detected Objects", combined_img)

        # Update GUI
        update_gui()

        # Exit loop on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # Release resources and close windows
    camera.stop()
    cv2.destroyAllWindows()
