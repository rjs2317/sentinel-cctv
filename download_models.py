"""OpenCV Zoo 얼굴 모델 다운로드 (Docker 빌드 중 1회 실행)"""
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

out = Path("models")
out.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dest = out / name
    print("download:", url)
    urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size
    # Git LFS 안내 파일(수백 바이트)이 받아지면 실패로 처리
    if size < 100_000:
        raise SystemExit(f"{name} 다운로드 실패 (크기 {size} bytes). 네트워크를 확인하세요.")
    print(f"  ok: {name} ({size:,} bytes)")
