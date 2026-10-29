"""
SentinelHub CCTV 범인 탐지 서버 (FastAPI + OpenCV + YOLO + 얼굴 인식)

- YOLO로 사람을 찾고, OpenCV 얼굴 인식(YuNet + SFace)으로 '등록된 범인'인지 판별
- 범인  → 빨간 박스 + 화면 빨간 테두리 + 경고음 + 로그
- 일반인 → 초록 박스, 경보 없음 (얼굴은 자동 블러 처리 = 개인정보 마스킹)
- 범인 등록은 CCTV 페이지의 '내 얼굴 등록' 버튼으로 (사진 원본은 저장하지 않고 특징값만 저장)

딜레이를 줄이기 위해 3개의 스레드로 나눠서 동작:
  1) 카메라 스레드 : 웹캠에서 계속 읽고 '가장 최신 화면'만 보관
  2) 화면 스레드   : 최신 화면에 마지막 AI 결과(박스·블러)를 그려서 바로 전송  → 영상은 실시간
  3) AI 스레드     : 자기 속도대로 사람·얼굴 판별 후 결과만 갱신             → 느려도 영상이 밀리지 않음

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

APP_VERSION = "v9 (2026-10-29) 범인 탐지 실시간 메시지"

# ───────── Prometheus 지표 (Grafana 관제 패널용) ─────────
M_ALERT = Gauge("cctv_alert_active", "범인 경보 상태 (1=범인 감지, 0=정상)")
M_PERSONS = Gauge("cctv_person_count", "현재 화면에서 탐지된 사람 수")
M_VISITORS = Gauge("cctv_visitor_count", "현재 화면의 일반인 수")
M_CAMERA = Gauge("cctv_camera_up", "카메라 신호 상태 (1=정상, 0=신호 없음)")
M_CAM_ENABLED = Gauge("cctv_camera_enabled", "카메라 켜짐 여부 (1=켜짐, 0=관리자가 끔)")
M_ENROLLED = Gauge("cctv_suspect_templates", "등록된 범인 얼굴 특징값 개수")
M_AI_FPS = Gauge("cctv_ai_fps", "AI 판별 속도 (초당 처리 횟수)")
M_INTRUSIONS = Counter("cctv_intrusions", "누적 범인 감지 횟수")

# ───────── 설정 (환경변수로 바꿀 수 있음) ─────────
BASE_DIR = Path(__file__).resolve().parent
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo26n.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.5"))
ALERT_HOLD_SEC = float(os.getenv("ALERT_HOLD_SEC", "2.0"))
LOG_PATH = Path(os.getenv("CCTV_LOG", str(BASE_DIR / "cctv_events.log")))

CAM_WIDTH = int(os.getenv("CAM_WIDTH", "1920"))        # 카메라에서 받는 해상도 (이 웹캠은 1920x1080만 지원)
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "1080"))
CAM_FPS = int(os.getenv("CAM_FPS", "15"))              # 카메라 초당 장수 (30/25/20/15/10/5 지원, 낮을수록 CPU 여유)
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "1280"))  # 받은 즉시 이 폭으로 줄여서 처리·전송 (끊김 방지)
AI_MAX_FPS = float(os.getenv("AI_MAX_FPS", "6"))       # AI 판별 최대 초당 횟수 (영상용 CPU 남겨두기)
AI_THREADS = int(os.getenv("AI_THREADS", "2"))         # AI가 쓰는 CPU 코어 수
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "88"))    # 화면 전송 화질 (60~95)
STREAM_FPS = int(os.getenv("STREAM_FPS", "15"))        # 브라우저로 보내는 초당 장수 (높을수록 부드럽지만 무거움)
AI_WIDTH = int(os.getenv("AI_WIDTH", "640"))           # AI는 이 폭으로 줄여서 판별 (화면 화질과 무관, 속도용)
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "320"))       # YOLO 입력 크기 (낮을수록 빠름)
CAMERA_ON_START = os.getenv("CAMERA_ON_START", "1") == "1"

FACE_DET_MODEL = os.getenv("FACE_DET_MODEL", str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"))
FACE_REC_MODEL = os.getenv("FACE_REC_MODEL", str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"))
FACE_DIR = Path(os.getenv("FACE_DIR", str(BASE_DIR / "faces")))
EMB_PATH = FACE_DIR / "suspect_embeddings.npy"
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.363"))  # SFace 코사인 유사도 권장 기준
CONFIRM_HITS = int(os.getenv("CONFIRM_HITS", "2"))               # 연속 N번 일치해야 범인 확정 (오탐 방지)
MIN_FACE = int(os.getenv("MIN_FACE", "30"))                      # 이보다 작은 얼굴은 판별하지 않음(AI 화면 기준 px)
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
    "seq": 0,                # 화면이 새로 만들어질 때마다 +1 (같은 화면을 중복 전송하지 않기 위해)
    "alert": False,
    "person_count": 0,
    "visitor_count": 0,
    "camera_ok": False,
    "resolution": "",
    "last_seen": 0.0,
    "last_event": None,
    "last_score": 0.0,
    "persons": [],           # AI 결과: 원본 화면 좌표의 사람 박스
    "faces": [],             # AI 결과: [(box, is_suspect, score)]
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


def face_box(face, frame_shape, scale=1.0):
    """YuNet 결과 → 원본 화면 좌표로 바꾸고 화면 안으로 자른 (x1, y1, x2, y2)"""
    h, w = frame_shape[:2]
    x, y, fw, fh = (float(v) * scale for v in face[:4])
    return max(int(x), 0), max(int(y), 0), min(int(x + fw), w), min(int(y + fh), h)


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
    t = max(8, w // 90)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, t)
    cv2.putText(frame, "SUSPECT DETECTED", (24, 24 + t * 3), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.8, w / 1280), RED, 3)


class FrameGrabber:
    """카메라를 별도 스레드에서 계속 읽고 '가장 최신 프레임'만 보관 (딜레이 방지)"""

    def __init__(self, enabled=True):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.ok = False
        self.enabled = enabled
        self.is_file = not CAMERA_SOURCE.isdigit()
        threading.Thread(target=self._run, daemon=True).start()

    def set_enabled(self, on: bool):
        with self.lock:
            self.enabled = on
            if not on:
                self.frame, self.ok = None, False

    def _open(self):
        if self.is_file:
            return cv2.VideoCapture(CAMERA_SOURCE)
        cap = cv2.VideoCapture(int(CAMERA_SOURCE), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(CAMERA_SOURCE))
        # 고해상도는 MJPG 형식이어야 웹캠이 제 속도를 내는 경우가 많음 (순서 중요: 형식 → 크기)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"camera resolution: {w}x{h} @ {fps:.0f}fps (requested {CAM_WIDTH}x{CAM_HEIGHT})", flush=True)
        with state_lock:
            if w > DISPLAY_WIDTH:
                state["resolution"] = f"{w}x{h} → {DISPLAY_WIDTH}x{int(h * DISPLAY_WIDTH / w)} @ {fps:.0f}fps"
            else:
                state["resolution"] = f"{w}x{h} @ {fps:.0f}fps"
        return cap

    def _run(self):
        cap = None
        while True:
            with self.lock:
                enabled = self.enabled
            if not enabled:
                if cap is not None:          # 카메라 장치를 실제로 놓아줌 (웹캠 불도 꺼짐)
                    cap.release()
                    cap = None
                time.sleep(0.3)
                continue
            if cap is None:
                cap = self._open()

            ok, frame = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with self.lock:
                    self.ok = False
                time.sleep(1)
                cap.release()
                cap = None
                continue
            fw = frame.shape[1]
            if fw > DISPLAY_WIDTH:   # 1080p를 받은 즉시 줄여두면 이후 모든 처리가 가벼워짐
                fh = int(frame.shape[0] * DISPLAY_WIDTH / fw)
                frame = cv2.resize(frame, (DISPLAY_WIDTH, fh), interpolation=cv2.INTER_AREA)
            with self.lock:
                if self.enabled:
                    self.frame, self.ok = frame, True
                    self.seq += 1
            if self.is_file:
                time.sleep(1 / 30)

    def read(self):
        """(성공 여부, 프레임 번호, 최신 프레임 사본)"""
        with self.lock:
            if not self.ok or self.frame is None:
                return False, self.seq, None
            return True, self.seq, self.frame.copy()


def ai_loop():
    """AI 스레드: 최신 화면을 가져와 사람·범인 판별 → 결과와 경보 상태만 갱신"""
    try:                                   # AI가 CPU를 독차지해서 영상이 끊기지 않도록 코어 수 제한
        import torch
        torch.set_num_threads(AI_THREADS)
    except Exception:
        pass
    model = YOLO(MODEL_PATH)
    engine = FaceEngine()
    hits = 0
    min_gap = 1.0 / AI_MAX_FPS if AI_MAX_FPS > 0 else 0
    last_run = 0.0
    last_seq = -1
    t_count, t_start = 0, time.time()

    while True:
        if not grabber.enabled:
            with state_lock:
                was_alert = state["alert"]
                state.update(alert=False, person_count=0, visitor_count=0, camera_ok=False,
                             last_seen=0.0, persons=[], faces=[])
            if was_alert:
                ev = write_event("suspect_cleared", subject=SUSPECT_ID, reason="camera_off")
                with state_lock:
                    state["last_event"] = ev
            for m in (M_ALERT, M_PERSONS, M_VISITORS, M_CAMERA, M_AI_FPS):
                m.set(0)
            hits = 0
            time.sleep(0.3)
            continue

        ok, seq, frame = grabber.read()
        if not ok:
            with state_lock:
                state["camera_ok"] = False
            M_CAMERA.set(0)
            time.sleep(0.2)
            continue
        if seq == last_seq:          # 새 화면이 아직 없으면 잠깐 대기
            time.sleep(0.005)
            continue
        wait = min_gap - (time.time() - last_run)
        if wait > 0:                 # AI 속도 제한 → 남는 CPU는 영상에 사용
            time.sleep(wait)
            continue
        last_seq = seq
        now = time.time()
        last_run = now

        # AI용으로 작게 줄이기 (화면 화질과 무관, 속도용)
        h, w = frame.shape[:2]
        scale = w / AI_WIDTH if w > AI_WIDTH else 1.0
        small = cv2.resize(frame, (int(w / scale), int(h / scale))) if scale > 1 else frame

        # 1) 사람 찾기
        result = model(small, classes=[PERSON_CLASS_ID], conf=CONFIDENCE, imgsz=YOLO_IMGSZ, verbose=False)[0]
        persons = [tuple(int(v * scale) for v in b.xyxy[0].tolist()) for b in result.boxes]

        # 2) 얼굴 판별 (사람이 있을 때만)
        with state_lock:
            templates = list(suspect_embeddings)
        faces = []
        if persons:
            for face in engine.detect(small):
                score = engine.best_score(engine.embed(small, face), templates)
                faces.append((face_box(face, frame.shape, scale), score >= MATCH_THRESHOLD, score))
        suspect_now = any(s for _, s, _ in faces)
        hits = hits + 1 if suspect_now else 0

        visitors = sum(1 for p in persons
                       if not any(s and center_inside(b, p) for b, s, _ in faces))

        # 3) 경보 상태
        with state_lock:
            if hits >= CONFIRM_HITS:
                state["last_seen"] = now
                state["last_score"] = max(sc for _, s, sc in faces if s)
            was_alert = state["alert"]
            is_alert = (now - state["last_seen"]) < ALERT_HOLD_SEC
            score = state["last_score"]
            state.update(alert=is_alert, persons=persons, faces=faces, person_count=len(persons),
                         visitor_count=visitors, camera_ok=True)

        # 4) 경보가 바뀌는 순간에만 로그
        event = None
        if is_alert and not was_alert:
            event = write_event("suspect_detected", subject=SUSPECT_ID,
                                score=round(score, 3), person_count=len(persons))
            M_INTRUSIONS.inc()
        elif was_alert and not is_alert:
            event = write_event("suspect_cleared", subject=SUSPECT_ID)
        if event:
            with state_lock:
                state["last_event"] = event

        M_ALERT.set(1 if is_alert else 0)
        M_PERSONS.set(len(persons))
        M_VISITORS.set(visitors)
        M_CAMERA.set(1)

        t_count += 1
        if now - t_start >= 2:
            M_AI_FPS.set(round(t_count / (now - t_start), 1))
            t_count, t_start = 0, now


def render_loop():
    """화면 스레드: 최신 카메라 화면 + 마지막 AI 결과를 그려서 JPEG로 만듦 (영상은 항상 실시간)"""
    last_seq = -1
    interval = 1.0 / max(STREAM_FPS, 1)
    while True:
        t0 = time.time()
        if not grabber.enabled:
            jpg = placeholder("CAMERA OFF")
            with state_lock:
                state["jpeg"] = jpg
                state["seq"] += 1
            time.sleep(0.5)
            continue

        ok, seq, frame = grabber.read()
        if not ok:
            with state_lock:
                state["jpeg"] = placeholder("CAMERA STARTING...")
                state["seq"] += 1
            time.sleep(0.5)
            continue
        if seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq

        with state_lock:
            persons = list(state["persons"])
            faces = list(state["faces"])
            is_alert = state["alert"]
            enrolled = len(suspect_embeddings)

        suspect_faces = [b for b, s, _ in faces if s]
        if BLUR_OTHERS:
            for b, s, _ in faces:
                if not s:
                    blur_region(frame, b)
        thick = max(2, frame.shape[1] // 400)
        font = max(0.6, frame.shape[1] / 1600)
        for p in persons:
            is_suspect = any(center_inside(fb, p) for fb in suspect_faces)
            color = RED if is_suspect else GREEN
            label = "SUSPECT" if is_suspect else "VISITOR"
            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), color, thick)
            cv2.putText(frame, label, (p[0], max(p[1] - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, font, color, 2)
        if not enrolled:
            cv2.putText(frame, "NO SUSPECT ENROLLED", (24, frame.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (0, 200, 255), 2)
        if is_alert:
            draw_alert_border(frame)

        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        with state_lock:
            state["jpeg"] = jpg
            state["seq"] += 1

        spent = time.time() - t0
        if spent < interval:
            time.sleep(interval - spent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global grabber
    suspect_embeddings.extend(load_embeddings())
    M_ENROLLED.set(len(suspect_embeddings))
    M_CAM_ENABLED.set(1 if CAMERA_ON_START else 0)
    with state_lock:
        state["jpeg"] = placeholder("LOADING AI MODELS...")
    grabber = FrameGrabber(enabled=CAMERA_ON_START)
    threading.Thread(target=render_loop, daemon=True).start()
    threading.Thread(target=ai_loop, daemon=True).start()
    print(f"SentinelHub CCTV {APP_VERSION}", flush=True)
    yield


app = FastAPI(title="SentinelHub CCTV", lifespan=lifespan)
enroll_lock = threading.Lock()
enroll_engine = None
grabber: "FrameGrabber | None" = None


def mjpeg_stream():
    """새 화면이 생겼을 때만 보냄 → 쌓여서 밀리는 딜레이 방지"""
    last = -1
    while True:
        with state_lock:
            seq, jpg = state["seq"], state["jpeg"]
        if jpg and seq != last:
            last = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        else:
            time.sleep(0.01)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/embed")
def embed():
    """Grafana 패널에 넣는 관제용 화면 (영상 + 경보 표시 + 자동 경보음)"""
    return FileResponse(BASE_DIR / "static" / "embed.html",
                        headers={"Cache-Control": "no-store"})


ALARM_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}


@app.get("/alarm-sound")
def alarm_sound():
    """static 폴더의 alarm.mp3 / alarm.wav / alarm.ogg / alarm.m4a 중 있는 파일을 경보음으로 사용"""
    for ext, mime in ALARM_TYPES.items():
        path = BASE_DIR / "static" / f"alarm{ext}"
        if path.exists():
            return FileResponse(path, media_type=mime, headers={"Cache-Control": "no-store"})
    return JSONResponse({"ok": False, "message": "static 폴더에 alarm.mp3 파일이 없습니다."}, status_code=404)


@app.get("/video")
def video():
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/status")
def status():
    with state_lock:
        return {
            "version": APP_VERSION,
            "alert": state["alert"],
            "person_count": state["person_count"],
            "visitor_count": state["visitor_count"],
            "camera_ok": state["camera_ok"],
            "camera_enabled": bool(grabber and grabber.enabled),
            "resolution": state["resolution"],
            "enrolled": len(suspect_embeddings),
            "last_event": state["last_event"],
        }


@app.post("/enroll")
def enroll():
    """지금 화면에 보이는 얼굴 1명을 범인으로 등록 (특징값만 저장)"""
    global enroll_engine
    ok, _, frame = grabber.read() if grabber else (False, 0, None)
    if not ok:
        return JSONResponse({"ok": False, "message": "카메라가 꺼져 있거나 영상이 아직 없습니다."}, status_code=400)
    h, w = frame.shape[:2]
    scale = w / AI_WIDTH if w > AI_WIDTH else 1.0
    small = cv2.resize(frame, (int(w / scale), int(h / scale))) if scale > 1 else frame   # 판별과 같은 크기로
    with enroll_lock:
        if enroll_engine is None:
            enroll_engine = FaceEngine()
        faces = enroll_engine.detect(small)
        if len(faces) != 1:
            return JSONResponse({"ok": False,
                                 "message": f"얼굴이 {len(faces)}개 보입니다. 1명만 정면으로 보이게 해주세요."},
                                status_code=400)
        feat = enroll_engine.embed(small, faces[0])
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


@app.post("/camera/{action}")
def camera_control(action: str):
    """카메라 켜기/끄기 (on / off). 켜고 끈 기록을 로그로 남김"""
    if action not in ("on", "off"):
        return JSONResponse({"ok": False, "message": "on 또는 off만 가능합니다."}, status_code=400)
    on = action == "on"
    if grabber.enabled != on:
        grabber.set_enabled(on)
        M_CAM_ENABLED.set(1 if on else 0)
        write_event("camera_on" if on else "camera_off")
    return {"ok": True, "camera_enabled": on}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
