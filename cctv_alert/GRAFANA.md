# Grafana CCTV 관제 대시보드 구성

## 데이터 흐름
```
CCTV 서버(:5001) ─ /metrics ─→ Prometheus ─→ Grafana  (지금 상태: 경보 ON/OFF, 사람 수, 카메라)
             └─ logs/cctv_events.log ─→ Alloy ─→ Loki ─→ Grafana  (기록: 언제 감지·해제됐나)
             └─ / (경보 페이지) ─────────────────────→ Grafana  (영상 + 빨간 테두리 + 경고음)
```

## 1. Prometheus가 CCTV 지표를 가져가게 하기
`/etc/prometheus/prometheus.yml`의 `scrape_configs:` 아래에 추가:
```yaml
  - job_name: "cctv"
    scrape_interval: 5s
    static_configs:
      - targets: ["localhost:5001"]
```
```bash
sudo systemctl restart prometheus
curl http://localhost:5001/metrics | grep cctv
```

| 지표 | 의미 |
|---|---|
| `cctv_alert_active` | 1 = 침입 경보 중, 0 = 정상 |
| `cctv_person_count` | 현재 화면 속 사람 수 |
| `cctv_camera_up` | 1 = 카메라 정상, 0 = 신호 없음 |
| `cctv_intrusions_total` | 누적 침입 감지 횟수 |

## 2. 대시보드 배치 (관제 화면)
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 침입 경보     │ 카메라 상태   │ 오늘 침입 횟수 │ 로그인 실패   │  ← 상태 신호등
├──────────────┴──────────────┼──────────────┴──────────────┤
│ CCTV 실시간 (경보 페이지)     │ 보안 이벤트 기록 (Logs)        │  ← 보고, 확인
├─────────────────────────────┴─────────────────────────────┤
│ 시간대별 침입 감지 / 로그인 실패 그래프                       │  ← 추세
└───────────────────────────────────────────────────────────┘
```

## 3. 패널별 설정

### 침입 경보 (Stat, Prometheus)
- 쿼리: `cctv_alert_active`
- Value mappings: `0 → 정상`, `1 → 침입 감지`
- Thresholds: Base 초록, `1` 빨강 / Color mode: Background
- 대시보드 새로고침: 5s

### 카메라 상태 (Stat, Prometheus)
- 쿼리: `cctv_camera_up`
- Value mappings: `0 → 신호 없음`(빨강), `1 → 정상`(초록)
- No value(서버 꺼짐)도 빨강으로: Standard options → No value `서버 꺼짐`

### 오늘 침입 횟수 (Stat, Prometheus)
- 쿼리: `increase(cctv_intrusions_total[$__range])`
- 시간 범위를 "Today so far"로 두면 오늘 횟수

### CCTV 실시간 (Text, HTML 모드)
```html
<iframe src="http://cctv.sentinelhub.com/" style="width:100%;height:100%;border:0"></iframe>
```
- 영상 속 빨간 박스·테두리는 서버가 영상에 직접 그리므로 그대로 보인다
- 경고음은 패널 안 "경보음 켜기"를 관제 시작 시 한 번 클릭
- HTML이 안 보이면 `grafana.ini`의 `[panels]`에 `disable_sanitize_html = true`
- 도메인 연결 전에는 `http://192.168.0.4:5001/`처럼 IP로

### 보안 이벤트 기록 (Logs, Loki) — STEP 08 이후
- 쿼리: `{job=~"cctv|fastapi"} | json`
- CCTV 침입과 로그인 실패를 한 화면에서 시간순으로 확인

### 시간대별 침입 감지 (Time series, Loki)
- 쿼리: `sum(count_over_time({job="cctv"} | json | event="intrusion_detected" [5m]))`

### 그래프에 침입 시점 표시 (Annotation)
- Dashboard settings → Annotations → Loki
- 쿼리: `{job="cctv"} | json | event="intrusion_detected"`
- CPU·네트워크 그래프 위에 침입 시각이 빨간 세로선으로 표시됨

## 4. Alloy에 CCTV 로그 추가 (STEP 08)
```alloy
local.file_match "cctv" {
  path_targets = [{"__path__" = "/home/<사용자>/sentinel-cctv/logs/cctv_events.log", "job" = "cctv"}]
}
loki.source.file "cctv" {
  targets    = local.file_match.cctv.targets
  forward_to = [loki.write.local.receiver]
}
```

## 5. 알림 (Grafana Alerting)
1. Alerting → Contact points → 새로 만들기 → Discord(Webhook URL) 또는 Email
2. Alerting → Alert rules → New
   - 쿼리: `cctv_alert_active`, 조건: `IS ABOVE 0`
   - Evaluate every `10s`, Pending period `0s`
   - 요약: `CCTV 침입 감지 - 관제 확인 필요`
3. 경보가 해제되면 자동으로 Resolved 알림

## 6. 관제 대응 절차 (발표·문서용)
| 단계 | 행동 | 근거 화면 |
|---|---|---|
| 탐지 | 침입 경보 빨강 + 경고음 + 알림 수신 | 상태 패널, 알림 |
| 확인 | 실시간 영상으로 오탐 여부 판단 | CCTV 패널 |
| 기록 | 감지·해제 시각 확인 | 이벤트 기록 패널 |
| 연관 분석 | 같은 시각 로그인 실패·네트워크 이상 여부 | 그래프 + Annotation |
| 보고 | 시각·지속시간·대응 결과 기록 | 로그 |
