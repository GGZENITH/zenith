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
from ttkthemes import ThemedStyle

from yolov8 import YOLOv8

# ------- CONFIG --------

# ----- Aim Config -----
MOVEMENT_AMP_X = 1
MOVEMENT_AMP_Y = 1
# Add these constants to control the two-step movement
INITIAL_MOVEMENT_THRESHOLD = 50  # Adjust this threshold based on your needs
FINE_ADJUSTMENT_THRESHOLD = 3   # Adjust this threshold based on your needs
FINE_ADJUSTMENT_AMP_X = 1  # Adjust fine adjustment amplification based on your needs
FINE_ADJUSTMENT_AMP_Y = 1    # Adjust fine adjustment amplification based on your needs

MOUSE_MOVEMENT_THRESHOLD = 1
TOGGLE_AIM_KEY = 'x'

# --- Triggerbot Config ---
TB_THRESHOLD = 0.2
TOGGLE_TB_KEY = 'z'
TB_HEAD_OFFSET = 50

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
''
# Ignore in the GUI
SCREEN_WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)
CONFIG_PATH = "config/default.json"
MODEL_PATH = "models/humans.onnx"
CONF_THRES = 0.5
IOU_THRES = 0.5
DETECTION_CLASS_ID = 0

# load all the default config files if they don't exist
model_filenames = ["apex.onnx", "cs2.onnx", "fortnite.onnx", "humans.onnx", "phantomforces.onnx", "roblox.onnx"]
model_urls = ["https://github.com/GGZENITH/zenith/raw/main/src/models/apex.onnx",
              "https://github.com/GGZENITH/zenith/raw/main/src/models/cs2.onnx",
              "https://github.com/GGZENITH/zenith/raw/main/src/models/fortnite.onnx",
              "https://github.com/GGZENITH/zenith/raw/main/src/models/humans.onnx",
              "https://github.com/GGZENITH/zenith/raw/main/src/models/phantomforces.onnx",
              "https://github.com/GGZENITH/zenith/raw/main/src/models/roblox.onnx"]
config_filenames = ["blatant.json", "cs2.json", "default.json", "headshotmachine.json", "legit.json"]
config_urls = ["https://github.com/GGZENITH/zenith/raw/main/src/config/blatant.json",
               "https://github.com/GGZENITH/zenith/raw/main/src/config/cs2.json",
               "https://github.com/GGZENITH/zenith/raw/main/src/config/default.json",
               "https://github.com/GGZENITH/zenith/raw/main/src/config/headshotmachine.json",
               "https://github.com/GGZENITH/zenith/raw/main/src/config/legit.json"]
asset_filenames = ["icon.png", "logo.png"]
asset_urls = ["https://github.com/GGZENITH/zenith/blob/main/src/assets/icon.png?raw=true",
              "https://github.com/GGZENITH/zenith/blob/main/src/assets/logo.png?raw=true"]

# Check if the model directory exists
if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("config"):
    os.makedirs("config")

if not os.path.exists("assets"):
    os.makedirs("assets")

# Combine model and config download loops
for filenames, urls, folder in \
        zip([model_filenames, config_filenames, asset_filenames],
            [model_urls, config_urls, asset_urls], ['models', 'config', "assets"]):
    for filename, url in zip(filenames, urls):
        file_path = os.path.join(folder, filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            # If not, download the file
            response = requests.get(url)

            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)

                print(f"File '{filename}' downloaded and saved to: {file_path}")
            else:
                print(f"Failed to download the file '{filename}'. HTTP status code: {response.status_code}")
                exit()
        else:
            print(f"File '{filename}' already exists at: {file_path}")


def save_config():
    config_data = {
        "Setup": {
            "WINDOW_WIDTH": WINDOW_WIDTH,
            "WINDOW_HEIGHT": WINDOW_HEIGHT,
            "MODEL_PATH": MODEL_PATH
        },
        "Aim": {
            "MOVEMENT_AMP_X": MOVEMENT_AMP_X,
            "MOVEMENT_AMP_Y": MOVEMENT_AMP_Y,
            "MOUSE_MOVEMENT_THRESHOLD": MOUSE_MOVEMENT_THRESHOLD,
            "TOGGLE_AIM_KEY": TOGGLE_AIM_KEY
        },
        "Triggerbot": {
            "TB_THRESHOLD": TB_THRESHOLD,
            "TOGGLE_TB_KEY": TOGGLE_TB_KEY,
            "TB_HEAD_OFFSET": TB_HEAD_OFFSET  # Add this line
        },
        "Debug": {
            # ... (Add other debug parameters)
        }
    }

    with open(CONFIG_PATH, "w") as config_file:
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

    current_model = os.path.basename(MODEL_PATH)
    config_gui.model_selection.set(current_model)

    # MAKE IT LOAD LAST CONFIG INTO GUI
    current_config = os.path.basename(CONFIG_PATH)
    config_gui.config_selection.set(current_config)


def load_config():
    try:
        with open(CONFIG_PATH, "r") as config_file:
            # Check if the file is empty
            if not config_file.read(1):
                raise ValueError("Empty file")

            config_file.seek(0)  # Reset file pointer to the beginning
            config_data = json.load(config_file)

            # Update variables with values from the loaded config
            global MOUSE_MOVEMENT_THRESHOLD, TOGGLE_AIM_KEY, MOVEMENT_AMP_X, MOVEMENT_AMP_Y
            global TB_THRESHOLD, TOGGLE_TB_KEY, TB_HEAD_OFFSET
            global WINDOW_WIDTH, WINDOW_HEIGHT, MODEL_PATH

            setup_config = config_data.get("Setup", {})
            WINDOW_WIDTH = setup_config.get("WINDOW_WIDTH", WINDOW_WIDTH)
            WINDOW_HEIGHT = setup_config.get("WINDOW_HEIGHT", WINDOW_HEIGHT)
            MODEL_PATH = setup_config.get("MODEL_PATH", MODEL_PATH)

            MOVEMENT_AMP_X = config_data.get("Aim", {}).get("MOVEMENT_AMP_X", MOVEMENT_AMP_X)
            MOVEMENT_AMP_Y = config_data.get("Aim", {}).get("MOVEMENT_AMP_Y", MOVEMENT_AMP_Y)
            MOUSE_MOVEMENT_THRESHOLD = config_data.get("Aim", {}).get("MOUSE_MOVEMENT_THRESHOLD",
                                                                      MOUSE_MOVEMENT_THRESHOLD)
            TOGGLE_AIM_KEY = config_data.get("Aim", {}).get("TOGGLE_AIM_KEY", TOGGLE_AIM_KEY)

            TB_THRESHOLD = config_data.get("Triggerbot", {}).get("TB_THRESHOLD", TB_THRESHOLD)
            TOGGLE_TB_KEY = config_data.get("Triggerbot", {}).get("TOGGLE_TB_KEY", TOGGLE_TB_KEY)
            TB_HEAD_OFFSET = config_data.get("Triggerbot", {}).get("TB_HEAD_OFFSET", TOGGLE_TB_KEY)

            # Update the GUI with the loaded configuration
            load_gui_config()

    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        load_gui_config()
        pass


def delete_entry(event):
    # Function to handle removing the placeholder text when the entry field is clicked
    event.widget.delete(0, "end")


def validate_key_entry(event):
    # Validate that only one character is entered and it is a valid key
    key = event.char.upper()  # Get the pressed key and convert to uppercase
    valid_keys = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Valid keys
    if len(key) == 1 and key in valid_keys:
        event.widget.delete(0, tk.END)


class ConfigGUI:
    def __init__(self, master):
        self.master = master
        master.title("Zenith")

        # Apply ThemedStyle for a modern look
        style = ThemedStyle(master)
        style.set_theme("equilux")  # You can choose a different theme

        # Configure label style
        style.configure("Bold.TLabel", font=("Helvetica", 12), background="#414141")

        master.configure(background=style.lookup("TFrame", "background"))

        # Load and display the image
        image_path = "assets/logo.png"
        zenith_image = tk.PhotoImage(file=image_path)

        image_label = ttk.Label(master, image=zenith_image)
        image_label.image = zenith_image  # Keep a reference to prevent image from being garbage collected
        image_label.place(relx=0.5, rely=0.12, anchor="center")

        # Draw a separator line
        separator_line = ttk.Separator(master, orient="horizontal")
        separator_line.place(relx=0.5, rely=0.25, anchor="center", relwidth=0.94)

        # Setup Config Section
        setup_frame = ttk.LabelFrame(master, text="Setup", padding=(15, 10))
        setup_frame.place(relx=0.03, rely=0.53, anchor="w", height=250)

        ttk.Label(setup_frame, text="Width:", style="Bold.TLabel").grid(row=0, column=0, sticky="w", pady=10)
        self.window_width_entry = ttk.Entry(setup_frame, style="Custom.TEntry", justify="right", font=("TkDefaultFont",
                                                                                                       12))
        self.window_width_entry.grid(row=0, column=1, pady=10, padx=(15, 0), sticky="e")

        ttk.Label(setup_frame, text="Height:", style="Bold.TLabel").grid(row=1, column=0, sticky="w", pady=10)
        self.window_height_entry = ttk.Entry(setup_frame, style="Custom.TEntry", justify="right", font=("TkDefaultFont",
                                                                                                        12))
        self.window_height_entry.grid(row=1, column=1, pady=10, padx=(15, 0), sticky="e")

        model_files = [f for f in os.listdir("models") if os.path.isfile(os.path.join("models", f))]
        ttk.Label(setup_frame, text="Model:", style="Bold.TLabel").grid(row=2, column=0, sticky="w", pady=10)
        self.model_selection = ttk.Combobox(setup_frame, values=model_files, state="readonly", style="Custom.TCombobox",
                                            font=("TkDefaultFont", 12))
        self.model_selection.grid(row=2, column=1, pady=10, padx=(15, 0), sticky="e")

        config_files = [f for f in os.listdir("config") if os.path.isfile(os.path.join("config", f))]
        ttk.Label(setup_frame, text="Config:", style="Bold.TLabel").grid(row=3, column=0, sticky="w", pady=10)
        self.config_selection = ttk.Combobox(setup_frame, values=config_files, state="readonly",
                                             style="Custom.TCombobox", font=("TkDefaultFont", 12))
        self.config_selection.grid(row=3, column=1, pady=10, padx=(15, 0), sticky="e")
        self.config_selection.bind("<<ComboboxSelected>>", self.load_selected_config)

        # Aim Config Section
        aim_frame = ttk.LabelFrame(master, text="Aim", padding=(15, 10))
        aim_frame.place(relx=0.4963, rely=0.53, anchor="center", height=250)

        ttk.Label(aim_frame, text="X Sens:", style="Bold.TLabel").grid(row=0, column=0, sticky="w", pady=10)
        self.mov_amp_x_entry = ttk.Entry(aim_frame, style="Custom.TEntry", justify="right", font=("TkDefaultFont", 12))
        self.mov_amp_x_entry.grid(row=0, column=1, pady=10, padx=(15, 0), sticky="e")

        ttk.Label(aim_frame, text="Y Sens:", style="Bold.TLabel").grid(row=1, column=0, sticky="w", pady=10)
        self.mov_amp_y_entry = ttk.Entry(aim_frame, style="Custom.TEntry", justify="right", font=("TkDefaultFont", 12))
        self.mov_amp_y_entry.grid(row=1, column=1, pady=10, padx=(15, 0), sticky="e")

        ttk.Label(aim_frame, text="Threshold:", style="Bold.TLabel").grid(row=2, column=0, sticky="w", pady=10)
        self.aim_thres = ttk.Entry(aim_frame, style="Custom.TEntry", justify="right", font=("TkDefaultFont", 12))
        self.aim_thres.grid(row=2, column=1, pady=10, padx=(15, 0), sticky="e")

        ttk.Label(aim_frame, text="Bind:", style="Bold.TLabel").grid(row=3, column=0, sticky="w", pady=10)
        self.toggle_aim_key_entry = ttk.Entry(aim_frame, style="Custom.TEntry", justify="right",
                                              font=("TkDefaultFont", 12))
        self.toggle_aim_key_entry.grid(row=3, column=1, pady=10, padx=(15, 0), sticky="e")
        self.toggle_aim_key_entry.bind("<FocusIn>", delete_entry)
        self.toggle_aim_key_entry.bind("<Key>", validate_key_entry)

        # Triggerbot Config Section
        tb_frame = ttk.LabelFrame(master, text="Triggerbot", padding=(15, 10))
        tb_frame.place(relx=0.97, rely=0.53, anchor="e", height=250)

        ttk.Label(tb_frame, text="Threshold:", style="Bold.TLabel").grid(row=0, column=0, sticky="w", pady=10)
        self.tb_threshold_entry = ttk.Entry(tb_frame, style="Custom.TEntry", justify="right",
                                            font=("TkDefaultFont", 12))
        self.tb_threshold_entry.grid(row=0, column=1, pady=10, padx=(15, 0), sticky="e")

        ttk.Label(tb_frame, text="Offset:", style="Bold.TLabel").grid(row=1, column=0, sticky="w", pady=10)
        self.tb_head_offset_entry = ttk.Entry(tb_frame, style="Custom.TEntry", justify="right",
                                              font=("TkDefaultFont", 12))
        self.tb_head_offset_entry.grid(row=1, column=1, pady=10, padx=(15, 0), sticky="e")

        ttk.Label(tb_frame, text="Bind:", style="Bold.TLabel").grid(row=2, column=0, sticky="w", pady=10)
        self.toggle_tb_key_entry = ttk.Entry(tb_frame, style="Custom.TEntry", justify="right",
                                             font=("TkDefaultFont", 12))
        self.toggle_tb_key_entry.grid(row=2, column=1, pady=10, padx=(15, 0), sticky="e")
        self.toggle_tb_key_entry.bind("<FocusIn>", delete_entry)
        self.toggle_tb_key_entry.bind("<Key>", validate_key_entry)

        # Apply Button
        style.configure('my.TButton', font=('Roboto Mono', 12))
        apply_button = ttk.Button(master, text="Apply Config", style="my.TButton", command=self.apply_config)
        apply_button.place(relx=0.5, rely=0.85, relwidth=0.94, relheight=0.1, anchor="center")

    def apply_config(self):
        global MOVEMENT_AMP_X, MOVEMENT_AMP_Y, MOUSE_MOVEMENT_THRESHOLD, TB_THRESHOLD, tb_enabled, yolov8_detector
        global TOGGLE_AIM_KEY, TB_HEAD_OFFSET, TOGGLE_TB_KEY, WINDOW_WIDTH, WINDOW_HEIGHT, MODEL_PATH

        # Retrieve values from GUI and apply them to the corresponding variables in your code
        WINDOW_WIDTH = int(self.window_width_entry.get())
        WINDOW_HEIGHT = int(self.window_height_entry.get())
        MODEL_PATH = os.path.join("models", self.model_selection.get())

        MOVEMENT_AMP_X = float(self.mov_amp_x_entry.get())
        MOVEMENT_AMP_Y = float(self.mov_amp_y_entry.get())
        MOUSE_MOVEMENT_THRESHOLD = float(self.aim_thres.get())
        TOGGLE_AIM_KEY = self.toggle_aim_key_entry.get()

        TB_THRESHOLD = float(self.tb_threshold_entry.get())
        TOGGLE_TB_KEY = self.toggle_tb_key_entry.get()
        TB_HEAD_OFFSET = float(self.tb_head_offset_entry.get())

        save_config()
        yolov8_detector = YOLOv8(MODEL_PATH, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
        cv2.destroyAllWindows()
        camera.stop()
        camera.start(region=((SCREEN_WIDTH - WINDOW_WIDTH) // 2, (SCREEN_HEIGHT - WINDOW_HEIGHT) // 2,
                             (SCREEN_WIDTH + WINDOW_WIDTH) // 2, (SCREEN_HEIGHT + WINDOW_HEIGHT) // 2))
        time.sleep(0.2)

    def load_selected_config(self, event):
        _ = event
        global CONFIG_PATH
        # Load the selected config whenever a new config path is selected
        selected_config_path = os.path.join("config", self.config_selection.get())
        if os.path.isfile(selected_config_path):
            CONFIG_PATH = os.path.join("config", self.config_selection.get())
            load_config()


# Create the main window and the configuration GUI
root = tk.Tk()
icon_image = tk.PhotoImage(file=os.path.abspath("assets/icon.png"))
root.iconphoto(True, icon_image)
config_gui = ConfigGUI(root)
root.geometry("1000x500")

# Load the configuration when the program starts
load_config()


def update_gui():
    root.update_idletasks()
    root.update()


# Initialize camera and frame capture
camera = dxcam.create(device_idx=0)
camera.start(region=((SCREEN_WIDTH - WINDOW_WIDTH) // 2, (SCREEN_HEIGHT - WINDOW_HEIGHT) // 2,
                     (SCREEN_WIDTH + WINDOW_WIDTH) // 2, (SCREEN_HEIGHT + WINDOW_HEIGHT) // 2))

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

        # Move the mouse cursor
        if closest_detection is not None and mouse_movement_enabled:
            # Calculate displacement
            displacement_x = closest_detection[0] - center_x
            displacement_y = closest_detection[1] - center_y

            # Check if the displacement exceeds the initial threshold for the first step
            if abs(displacement_x) > INITIAL_MOVEMENT_THRESHOLD or abs(displacement_y) > INITIAL_MOVEMENT_THRESHOLD:
                # Perform the initial movement
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(displacement_x * MOVEMENT_AMP_X),
                                     int(displacement_y * MOVEMENT_AMP_Y), 0, 0)
            elif abs(displacement_x) > FINE_ADJUSTMENT_THRESHOLD or abs(displacement_y) > FINE_ADJUSTMENT_THRESHOLD:
                # Perform fine adjustments with a separate threshold and amplification
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(displacement_x * FINE_ADJUSTMENT_AMP_X),
                                     int(displacement_y * FINE_ADJUSTMENT_AMP_Y), 0, 0)
            else:
                # If the cursor is within both thresholds, stop mouse movement
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, 0, 0, 0)

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
        waitKey = cv2.waitKey(1) & 0xFF
        if waitKey == ord('q'):
            break

finally:
    # Release resources and close windows
    camera.stop()
    cv2.destroyAllWindows()
