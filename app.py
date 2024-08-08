import queue
import cv2
import streamlit as st
import pandas as pd
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from turn import get_ice_servers
import threading
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import numpy as np
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

with open("config/coco.yaml", "r") as file:
    data = file.read()

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

lock = threading.Lock()
img_container = {"img": None}

model = YOLO("yolov10n.pt")

def draw_corner_lines(image, x1, y1, x2, y2, color, thickness):
    length = (x2 - x1) // 5
    cv2.line(image, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - length), color, thickness)

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 255), thickness=1, padding=10):
    # Tính kích thước văn bản
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    text_x = x
    text_y = y - baseline

    # Vẽ hình chữ nhật bao quanh văn bản
    top_left = (text_x, text_y - padding)
    bottom_right = (text_x + text_width + 2 * padding, text_y + text_height + padding)
    cv2.rectangle(img, top_left, bottom_right, color, -1)  # Vẽ hình chữ nhật với màu nền trắng

    cv2.putText(img, text, position, font, 1, (0, 255, 255), 2)

def get_color_for_label(label):
    idx = class_names.index(label)
    return tuple(colors[idx])

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    results = model.predict(img)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()  # added this line
    px = pd.DataFrame(a).astype("float")
    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        if row[4] > 0.45:
            if d == 67:
                cv2.putText(
                    img,
                    "Detected cell phone !!!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            label_name = data.split("\n")[d].split(":")[1].strip()
            color = get_color_for_label(label_name)
            draw_corner_lines(img, x1, y1, x2, y2, color, 2)
            draw_text_with_background(img, label_name, (x1, y1),font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=color, thickness=1, padding=5)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# webrtc_ctx = webrtc_streamer(
#     key="object-detection",
#     video_frame_callback=video_frame_callback,
# )


st.markdown("This demo of GST.GDX ")
