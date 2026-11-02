# SentinelHub CCTV — 블랙리스트 탐지·얼굴 마스킹

SentinelHub 팀 프로젝트(AI 로그 개인정보 탐지·마스킹·모니터링)의 **물리 보안 확장 파트**입니다.
웹캠 영상에서 등록된 인물(블랙리스트)을 찾아 경보하고, 일반인 얼굴은 자동으로 블러 처리하며,
이벤트 로그를 Loki로 보내 팀 Grafana 대시보드에서 관제합니다.

> 내부망 실습용입니다. 촬영과 얼굴 등록은 **동의한 팀원만** 대상으로 합니다.

## 구성
```
노트북 웹캠 ──(VirtualBox)──> Kali 192.168.0.108
  ├─ cctv 컨테이너 (FastAPI, :5001)  영상 처리 · AI 판별 · 블러 · 경보 · 로그
  └─ loki 컨테이너 (:3101)            CCTV 이벤트 로그 저장
         │
         └──> 팀원 PC Grafana 192.168.0.4:3000
               CCTV 패널(/embed) · Logs · Detail_Logs · 경보 규칙(🔴 Firing)
```

판별 순서: **YOLO**(사람) → **YuNet**(얼굴) → **SFace**(등록 얼굴과 코사인 유사도 비교)
→ 기준값 이상이 연속 2번이면 블랙리스트 확정.

## 실행 (Kali)
```bash
git clone https://github.com/rjs2317/sentinel-cctv.git
cd sentinel-cctv
ls /dev/video0                     # 웹캠 확인 (없으면 VirtualBox 장치 메뉴에서 연결)
sudo docker compose up -d --build
```
- 관리자 페이지: `http://<Kali IP>:5001`
- Grafana 패널용 관제 화면: `http://<Kali IP>:5001/embed`
- Loki 상태: `curl -v http://localhost:3101/ready`
- 상태 확인: `sudo docker compose ps` / 에러: `sudo docker compose logs --tail 20`

## 코드 수정 후 반영
```bash
# Windows (Git Bash)
git add . && git commit -m "수정 내용" && git push
# Kali
git pull && sudo docker compose up -d --build   # 설정값(docker-compose.yml)만 바꿨으면 --build 생략 가능
```

## 주요 파일
| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI 서버: 카메라 입력, 탐지, 블러, 경보 상태, 로그 기록·Loki 전송, 지표 |
| `static/index.html` | 관리자 페이지: 카메라 켜기/끄기, 블랙리스트 등록(카메라·사진)·초기화, 상태 |
| `static/embed.html` | Grafana 패널용 관제 화면: 빨간 테두리, 경보 메시지, 경보음 |
| `download_models.py` | 얼굴 모델(YuNet·SFace) 내려받기 |
| `Dockerfile` / `docker-compose.yml` | 실행 환경, 포트·웹캠·설정값·Loki 컨테이너 |
| `GRAFANA.md` | Grafana 연동 참고 |

## 블랙리스트 등록
**카메라로 등록**
1. 관리자 페이지에서 카메라를 켜고 대상자 1명만 화면에 나오게 한다.
2. **📸 지금 얼굴을 블랙리스트에 등록**을 정면·좌우로 3~5번 누른다.

**사진 파일로 등록 (v13)** — 대상이 카메라 앞에 없을 때
1. **등록 대상의 동의를 받은 사진입니다**를 체크한다 (체크가 없으면 서버가 거절).
2. **🖼 사진 파일로 블랙리스트 등록** → jpg/png(10MB 이하), 얼굴이 1개인 사진만.
3. 사진에서 특징값만 뽑고 사진은 저장하지 않는다. 로그에 `method=photo`로 남는다.

실습이 끝나면 **🗑 등록 초기화**로 특징값을 삭제하고 `ls faces/`로 `.npy`가 없는지 확인한다.

## 탐지 기준 (docker-compose.yml)
| 설정 | 값 | 의미 |
|---|---|---|
| `MATCH_THRESHOLD` | 0.7 | 얼굴 유사도가 이 값 이상이면 블랙리스트 (OpenCV 권장 0.363, 높일수록 오탐↓ 미탐↑) |
| `CONFIRM_HITS` | 2 | 연속 몇 번 일치해야 경보할지 |
| `ALERT_HOLD_SEC` | 10 | 얼굴이 잠깐 안 보여도 이 시간(초)은 같은 등장으로 보고 경보 유지 |
| `BLUR_OTHERS` | 1 | 일반인 얼굴 블러 (끄지 않음) |
| `AI_WIDTH` / `AI_MAX_FPS` | 640 / 6 | AI 판별용 축소 폭 / 초당 판별 횟수 |

## 로그 형식 (`logs/cctv_events.log`, Loki `job="sentinelhub-cctv"`)
```json
{"time": "2026-11-02T15:00:34+09:00", "source": "cctv", "host": "192.168.0.108", "event": "suspect_detected", "subject": "blacklist-01", "score": 0.761, "person_count": 1}
```
| event | 뜻 |
|---|---|
| `suspect_detected` / `suspect_cleared` | 블랙리스트 감지 / 경보 해제 |
| `camera_on` / `camera_off` | 관리자가 카메라를 켜거나 끔 |
| `suspect_enrolled` | 블랙리스트 등록 (`method`: camera / photo) |
| `suspect_enrollment_reset` | 등록 초기화 |
| `server_start` | 서버 시작 (버전 포함) |

## 개인정보 보호
- 일반인 얼굴은 서버에서 가우시안 블러 후 송출 (원본 영상 미저장)
- 얼굴 사진 대신 특징값(`faces/suspect_embeddings.npy`)만 로컬 저장
- 로그에는 실명 대신 가명 ID(`blacklist-01`), 시각·점수·인원 수만
- 특징값·로그·인증서 개인키는 GitHub에 올리지 않음 (`.gitignore`)

## 버전 이력
| 날짜 | 버전 | 내용 |
|---|---|---|
| 11/02 | 설정 | 기준값 0.363 → 0.7, 경보 유지 2초 → 10초 |
| 10/30 | v13 | 사진 파일로 블랙리스트 등록 (동의 확인, 사진 미저장, 등록 방법 로그) |
| 10/30 | v12 | '범인' → '블랙리스트' 문구 통일 |
| 10/30 | v11 | Kali에 Loki 컨테이너 추가(외부 3101), Grafana Logs·Detail_Logs 연동 |
| 10/29 | v10 | Loki 로그 전송, 전송 상태 표시 |
| 10/29 | v7~v9 | Grafana용 관제 화면, 화면 배치, 실시간 경보 메시지 |
| 10/28 | v6 | 사용자 경보음 |
| 10/27 | v4~v5 | 실시간 화면 구조(스레드 3개), 카메라 켜기/끄기, 끊김 개선 |
| 10/27 | 얼굴 인식 | 등록 인물만 경보, 일반인 블러 |
| 10/26 | 첫 버전 | 사람 침입 감지 (YOLO) |
