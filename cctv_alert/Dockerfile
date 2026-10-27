# SentinelHub CCTV 범인 탐지 서버
FROM python:3.12-slim

# OpenCV가 화면 처리에 필요로 하는 시스템 라이브러리
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 전용 PyTorch를 먼저 설치 (GPU 버전보다 훨씬 가벼움)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# AI 모델을 이미지 안에 미리 받아두기
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# 얼굴 찾기(YuNet) / 얼굴 비교(SFace) 모델 - OpenCV Zoo
COPY download_models.py .
RUN python download_models.py

COPY main.py .
COPY static ./static

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
