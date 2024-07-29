import threading
from ultralytics import YOLO
import cv2
import pandas as pd
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from turn import get_ice_servers

with open("config/coco.yaml", "r") as file:
    data = file.read()


# lock = threading.Lock()
# img_container = {"img": None}

model = YOLO("yolov8s.pt")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # with lock:
    #     img_container["img"] = img
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
        if row[4] > 0.35:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_name = data.split("\n")[d].split(":")[1].strip()
            cv2.putText(
                img,
                str(label_name),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            
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

