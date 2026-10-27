# SentinelHub CCTV 침입 감지

웹캠 영상에서 YOLO로 사람을 찾으면 빨간 박스·빨간 테두리를 그리고, 브라우저 페이지에서 경고음을 울리는 FastAPI 서버.

## 구조
```
Windows (Git Bash) --git push--> GitHub --git clone/pull--> Kali --docker compose--> 실행
```

## Kali에서 실행
```bash
git clone https://github.com/<내아이디>/sentinel-cctv.git
cd sentinel-cctv
sudo docker compose up -d --build
```
- 접속: `http://<Kali IP>:5001` → "경보음 켜기" 클릭
- 로그: `logs/cctv_events.log`
- 화면 로그 보기: `sudo docker compose logs -f`
- 끄기: `sudo docker compose down`

## 코드 수정 후 반영
Windows(Git Bash):
```bash
git add . && git commit -m "수정 내용" && git push
```
Kali:
```bash
git pull && sudo docker compose up -d --build
```

## 주요 파일
| 파일 | 역할 |
|---|---|
| `main.py` | 영상 처리·사람 탐지·경보 상태·로그 기록 |
| `static/index.html` | CCTV 화면, 빨간 깜빡임, 경고음 |
| `Dockerfile` | 실행 환경 포장 설명서 |
| `docker-compose.yml` | 포트·웹캠·로그 폴더 연결 |

> 내부망 실습용. 카메라에는 동의한 팀원만 촬영한다.

## 범인 지정 (얼굴 인식)
1. CCTV 페이지(`http://<Kali IP>:5001`)에서 범인 역할이 카메라 정면에 혼자 선다
2. **📸 지금 얼굴을 범인으로 등록** 클릭 (정면·약간 옆 등 3~5회)
3. 등록된 얼굴 → 빨간 `SUSPECT` + 경보, 그 외 사람 → 초록 `VISITOR` + 얼굴 블러
- 사진 원본은 저장하지 않고 얼굴 특징값(`faces/suspect_embeddings.npy`)만 저장, GitHub에 올리지 않음
- 로그에는 이름 대신 `suspect-01` 가명 ID
- 판별이 너무 예민/둔하면 `docker-compose.yml`의 `MATCH_THRESHOLD` 조정
- 실습 종료 시 **🗑 등록 초기화**로 특징값 삭제
