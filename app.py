import queue
import cv2
import streamlit as st
import pandas as pd
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from turn import get_ice_servers
from yolov8 import YOLOv8

model_path = "yolov8s.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.35, iou_thres=0.45)

with open("config/coco.yaml", "r") as file:
    data = file.read()

# lock = threading.Lock()
# img_container = {"img": None}


result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # with lock:
    #     img_container["img"] = img
    boxes, scores, labels = yolov8_detector(img)
    combine_img = yolov8_detector.draw_detections(img)
    for label in labels:
        if label == 67:
            cv2.putText(combine_img, "Detected cell phone", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
    return av.VideoFrame.from_ndarray(combine_img, format="bgr24")

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

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown(
    "This demo of GST.GDX "
)