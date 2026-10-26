"""
SentinelHub CCTV 침입 감지 서버 (FastAPI + OpenCV + YOLO)

- 웹캠 영상에서 YOLO로 '사람'을 찾으면 빨간 박스 + 빨간 화면 테두리 표시
- /status 로 경보 상태를 알려주고, 브라우저 페이지가 경고음을 울림
- 경보 시작/해제 시 JSON 로그를 남김 (나중에 Loki → Grafana 연동용)

실행:  uvicorn main:app --host 0.0.0.0 --port 5001
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from ultralytics import YOLO

# ───────── Prometheus 지표 (Grafana 관제 패널용) ─────────
M_ALERT = Gauge("cctv_alert_active", "침입 경보 상태 (1=경보, 0=정상)")
M_PERSONS = Gauge("cctv_person_count", "현재 화면에서 탐지된 사람 수")
M_CAMERA = Gauge("cctv_camera_up", "카메라 신호 상태 (1=정상, 0=신호 없음)")
M_INTRUSIONS = Counter("cctv_intrusions", "누적 침입 감지 횟수")

# ───────── 설정 (환경변수로 바꿀 수 있음) ─────────
BASE_DIR = Path(__file__).resolve().parent
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")            # 웹캠 번호(0) 또는 동영상 파일 경로
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo26n.pt")         # 처음 실행 시 자동 다운로드
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.5"))         # 이 확률 이상이면 '사람'으로 인정
ALERT_HOLD_SEC = float(os.getenv("ALERT_HOLD_SEC", "2.0")) # 사라진 뒤 몇 초간 경보 유지
LOG_PATH = Path(os.getenv("CCTV_LOG", str(BASE_DIR / "cctv_events.log")))

PERSON_CLASS_ID = 0          # YOLO(COCO) 기준 0번 = person
RED = (0, 0, 255)            # OpenCV는 BGR 순서

# ───────── 공유 상태 ─────────
state_lock = threading.Lock()
state = {
    "jpeg": None,
    "alert": False,
    "person_count": 0,
    "camera_ok": False,
    "last_seen": 0.0,
    "last_event": None,
}


def write_event(event: str, **fields):
    """보안 이벤트를 JSON 한 줄로 기록"""
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "cctv",
        "event": event,
        **fields,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(record, flush=True)
    return record


def placeholder(text: str) -> bytes:
    """카메라가 없을 때 보여줄 검은 화면"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def draw_boxes(frame, boxes):
    """탐지된 사람마다 빨간 박스"""
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(frame, f"SUSPECT {conf:.0%}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)


def draw_alert_border(frame):
    """화면 전체 빨간 테두리 + 경고 문구"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 14)
    cv2.putText(frame, "INTRUDER DETECTED", (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)


def open_source():
    src = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    return cv2.VideoCapture(src), not CAMERA_SOURCE.isdigit()


def capture_loop():
    with state_lock:
        state["jpeg"] = placeholder("LOADING AI MODEL...")
    model = YOLO(MODEL_PATH)
    cap, is_file = open_source()

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:                       # 동영상 파일이면 처음부터 반복 재생
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with state_lock:
                state["camera_ok"] = False
                state["jpeg"] = placeholder("NO CAMERA SIGNAL")
            M_CAMERA.set(0)
            time.sleep(1)
            cap.release()
            cap, is_file = open_source()
            continue

        # 1) AI로 사람 찾기
        result = model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE,
                       imgsz=416, verbose=False)[0]
        boxes = []
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2, float(b.conf[0])))

        # 2) 경보 상태 계산 (깜빡임 방지를 위해 ALERT_HOLD_SEC 동안 유지)
        now = time.time()
        with state_lock:
            if boxes:
                state["last_seen"] = now
            was_alert = state["alert"]
            is_alert = (now - state["last_seen"]) < ALERT_HOLD_SEC

        # 3) 화면에 그리기
        if boxes:
            draw_boxes(frame, boxes)
        if is_alert:
            draw_alert_border(frame)
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

        # 4) 경보가 바뀌는 순간에만 로그 기록
        event = None
        if is_alert and not was_alert:
            event = write_event("intrusion_detected", person_count=len(boxes))
            M_INTRUSIONS.inc()
        elif was_alert and not is_alert:
            event = write_event("intrusion_cleared")

        M_ALERT.set(1 if is_alert else 0)
        M_PERSONS.set(len(boxes))
        M_CAMERA.set(1)

        with state_lock:
            state.update(jpeg=jpeg, alert=is_alert, person_count=len(boxes), camera_ok=True)
            if event:
                state["last_event"] = event


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=capture_loop, daemon=True).start()
    yield


app = FastAPI(title="SentinelHub CCTV", lifespan=lifespan)


def mjpeg_stream():
    while True:
        with state_lock:
            jpg = state["jpeg"]
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.05)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/video")
def video():
    return StreamingResponse(mjpeg_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    with state_lock:
        return {
            "alert": state["alert"],
            "person_count": state["person_count"],
            "camera_ok": state["camera_ok"],
            "last_event": state["last_event"],
        }


@app.get("/metrics")
def metrics():
    """Prometheus가 주기적으로 가져가는 지표"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
