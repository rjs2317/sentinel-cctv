"""
SentinelHub CCTV 범인 탐지 서버 (FastAPI + OpenCV + YOLO + 얼굴 인식)

- YOLO로 사람을 찾고, OpenCV 얼굴 인식(YuNet + SFace)으로 '등록된 범인'인지 판별
- 범인  → 빨간 박스 + 화면 빨간 테두리 + 경고음 + 로그
- 일반인 → 초록 박스, 경보 없음 (얼굴은 자동 블러 처리 = 개인정보 마스킹)
- 범인 등록은 CCTV 페이지의 '내 얼굴 등록' 버튼으로 (사진 원본은 저장하지 않고 특징값만 저장)

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
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from ultralytics import YOLO

# ───────── Prometheus 지표 (Grafana 관제 패널용) ─────────
M_ALERT = Gauge("cctv_alert_active", "범인 경보 상태 (1=범인 감지, 0=정상)")
M_PERSONS = Gauge("cctv_person_count", "현재 화면에서 탐지된 사람 수")
M_VISITORS = Gauge("cctv_visitor_count", "현재 화면의 일반인 수")
M_CAMERA = Gauge("cctv_camera_up", "카메라 신호 상태 (1=정상, 0=신호 없음)")
M_ENROLLED = Gauge("cctv_suspect_templates", "등록된 범인 얼굴 특징값 개수")
M_INTRUSIONS = Counter("cctv_intrusions", "누적 범인 감지 횟수")

# ───────── 설정 (환경변수로 바꿀 수 있음) ─────────
BASE_DIR = Path(__file__).resolve().parent
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo26n.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.5"))
ALERT_HOLD_SEC = float(os.getenv("ALERT_HOLD_SEC", "2.0"))
LOG_PATH = Path(os.getenv("CCTV_LOG", str(BASE_DIR / "cctv_events.log")))

FACE_DET_MODEL = os.getenv("FACE_DET_MODEL", str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"))
FACE_REC_MODEL = os.getenv("FACE_REC_MODEL", str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"))
FACE_DIR = Path(os.getenv("FACE_DIR", str(BASE_DIR / "faces")))
EMB_PATH = FACE_DIR / "suspect_embeddings.npy"
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.363"))  # SFace 코사인 유사도 권장 기준
RECOG_EVERY = int(os.getenv("RECOG_EVERY", "3"))                 # N프레임마다 얼굴 인식 (속도용)
CONFIRM_HITS = int(os.getenv("CONFIRM_HITS", "2"))               # 연속 N번 일치해야 범인 확정 (오탐 방지)
MIN_FACE = int(os.getenv("MIN_FACE", "40"))                      # 이보다 작은 얼굴은 판별하지 않음(px)
BLUR_OTHERS = os.getenv("BLUR_OTHERS", "1") == "1"               # 일반인 얼굴 블러 처리
SUSPECT_ID = os.getenv("SUSPECT_ID", "suspect-01")               # 로그에는 이름 대신 가명 ID

PERSON_CLASS_ID = 0
RED = (0, 0, 255)
GREEN = (80, 200, 80)
FR_COSINE = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)

# ───────── 공유 상태 ─────────
state_lock = threading.Lock()
state = {
    "jpeg": None,
    "raw": None,             # 등록용 원본 프레임 (블러·박스 그리기 전)
    "alert": False,
    "person_count": 0,
    "visitor_count": 0,
    "camera_ok": False,
    "last_seen": 0.0,
    "last_event": None,
    "last_score": 0.0,
}
suspect_embeddings: list[np.ndarray] = []


class FaceEngine:
    """얼굴 찾기(YuNet) + 얼굴 특징값 비교(SFace)"""

    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(FACE_DET_MODEL, "", (320, 320), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(FACE_REC_MODEL, "")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        if faces is None:
            return []
        return [f for f in faces if f[2] >= MIN_FACE and f[3] >= MIN_FACE]

    def embed(self, frame, face):
        aligned = self.rec.alignCrop(frame, face)
        return self.rec.feature(aligned).astype(np.float32).reshape(1, -1).copy()

    def best_score(self, feat, templates):
        if not templates:
            return 0.0
        return max(float(self.rec.match(feat, t, FR_COSINE)) for t in templates)


def load_embeddings():
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    if EMB_PATH.exists():
        arr = np.load(EMB_PATH)
        return [row.reshape(1, -1).astype(np.float32) for row in arr]
    return []


def save_embeddings(items):
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    if items:
        np.save(EMB_PATH, np.vstack(items))
    elif EMB_PATH.exists():
        EMB_PATH.unlink()


def write_event(event: str, **fields):
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
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def face_box(face, frame_shape):
    """YuNet 결과 → 화면 안으로 자른 (x1, y1, x2, y2)"""
    h, w = frame_shape[:2]
    x, y, fw, fh = map(int, face[:4])
    return max(x, 0), max(y, 0), min(x + fw, w), min(y + fh, h)


def blur_region(frame, box):
    x1, y1, x2, y2 = box
    if x2 - x1 > 4 and y2 - y1 > 4:
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)


def center_inside(box, person):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def draw_alert_border(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 14)
    cv2.putText(frame, "SUSPECT DETECTED", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)


def open_source():
    src = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    return cv2.VideoCapture(src), not CAMERA_SOURCE.isdigit()


def capture_loop():
    with state_lock:
        state["jpeg"] = placeholder("LOADING AI MODELS...")
    model = YOLO(MODEL_PATH)
    engine = FaceEngine()
    cap, is_file = open_source()

    frame_idx = 0
    hits = 0
    faces_cache = []   # [(box, is_suspect, score)]

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
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

        raw = frame.copy()
        now = time.time()

        # 1) 사람 찾기 (YOLO)
        result = model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE, imgsz=416, verbose=False)[0]
        persons = [tuple(map(int, b.xyxy[0].tolist())) for b in result.boxes]

        # 2) 얼굴 인식 (N프레임마다, 사람이 있을 때만)
        with state_lock:
            templates = list(suspect_embeddings)
        if frame_idx % RECOG_EVERY == 0:
            faces_cache = []
            if persons:
                for face in engine.detect(frame):
                    score = engine.best_score(engine.embed(frame, face), templates)
                    faces_cache.append((face_box(face, frame.shape), score >= MATCH_THRESHOLD, score))
            suspect_now = any(s for _, s, _ in faces_cache)
            hits = hits + 1 if suspect_now else 0
            if hits >= CONFIRM_HITS:
                with state_lock:
                    state["last_seen"] = now
                    state["last_score"] = max(sc for _, s, sc in faces_cache if s)
        frame_idx += 1

        # 3) 경보 상태 (범인 확정 후 ALERT_HOLD_SEC 동안 유지)
        with state_lock:
            was_alert = state["alert"]
            is_alert = (now - state["last_seen"]) < ALERT_HOLD_SEC
            score = state["last_score"]

        # 4) 그리기: 일반인 얼굴 블러 → 사람 박스 → 테두리
        suspect_faces = [b for b, s, _ in faces_cache if s]
        if BLUR_OTHERS:
            for b, s, _ in faces_cache:
                if not s:
                    blur_region(frame, b)
        visitors = 0
        for p in persons:
            is_suspect = any(center_inside(fb, p) for fb in suspect_faces)
            color = RED if is_suspect else GREEN
            label = "SUSPECT" if is_suspect else "VISITOR"
            if not is_suspect:
                visitors += 1
            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), color, 3)
            cv2.putText(frame, label, (p[0], max(p[1] - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if not templates:
            cv2.putText(frame, "NO SUSPECT ENROLLED", (24, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if is_alert:
            draw_alert_border(frame)
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

        # 5) 경보가 바뀌는 순간에만 로그
        event = None
        if is_alert and not was_alert:
            event = write_event("suspect_detected", subject=SUSPECT_ID,
                                score=round(score, 3), person_count=len(persons))
            M_INTRUSIONS.inc()
        elif was_alert and not is_alert:
            event = write_event("suspect_cleared", subject=SUSPECT_ID)

        M_ALERT.set(1 if is_alert else 0)
        M_PERSONS.set(len(persons))
        M_VISITORS.set(visitors)
        M_CAMERA.set(1)
        with state_lock:
            state.update(jpeg=jpeg, raw=raw, alert=is_alert, person_count=len(persons),
                         visitor_count=visitors, camera_ok=True)
            if event:
                state["last_event"] = event


@asynccontextmanager
async def lifespan(app: FastAPI):
    suspect_embeddings.extend(load_embeddings())
    M_ENROLLED.set(len(suspect_embeddings))
    threading.Thread(target=capture_loop, daemon=True).start()
    yield


app = FastAPI(title="SentinelHub CCTV", lifespan=lifespan)
enroll_lock = threading.Lock()
enroll_engine = None


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
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    with state_lock:
        return {
            "alert": state["alert"],
            "person_count": state["person_count"],
            "visitor_count": state["visitor_count"],
            "camera_ok": state["camera_ok"],
            "enrolled": len(suspect_embeddings),
            "last_event": state["last_event"],
        }


@app.post("/enroll")
def enroll():
    """지금 화면에 보이는 얼굴 1명을 범인으로 등록 (특징값만 저장)"""
    global enroll_engine
    with state_lock:
        raw = None if state["raw"] is None else state["raw"].copy()
    if raw is None:
        return JSONResponse({"ok": False, "message": "카메라 영상이 아직 없습니다."}, status_code=400)
    with enroll_lock:
        if enroll_engine is None:
            enroll_engine = FaceEngine()
        faces = enroll_engine.detect(raw)
        if len(faces) != 1:
            return JSONResponse({"ok": False,
                                 "message": f"얼굴이 {len(faces)}개 보입니다. 1명만 정면으로 보이게 해주세요."},
                                status_code=400)
        feat = enroll_engine.embed(raw, faces[0])
    with state_lock:
        suspect_embeddings.append(feat)
        save_embeddings(suspect_embeddings)
        n = len(suspect_embeddings)
    M_ENROLLED.set(n)
    write_event("suspect_enrolled", subject=SUSPECT_ID, templates=n)
    return {"ok": True, "templates": n}


@app.post("/enroll/reset")
def enroll_reset():
    """등록된 범인 특징값 전부 삭제"""
    with state_lock:
        suspect_embeddings.clear()
        save_embeddings(suspect_embeddings)
    M_ENROLLED.set(0)
    write_event("suspect_enrollment_reset", subject=SUSPECT_ID)
    return {"ok": True, "templates": 0}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
