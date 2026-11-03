# 가벼운 4K 채보 엔진 — 집에서 학습하기

박 정렬 멜(한 박 48칸) + 작은 트랜스포머(3.4M 파라미터, fp16 약 6.8MB)로 2박씩 악보를 만든다.
구조는 GOCT 논문(Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts, MIT 코드) 방식을 새로 짠 것.

| 파일 | 하는 일 |
|---|---|
| `extract.py` | **데이터 받기**: 허깅페이스 osu-beatmaps 조각을 받아 4K mania 곡만 골라 학습 자료로 저장 |
| `model.py` | **모델** 정의 |
| `data.py` | 학습 샘플 만들기 (`train.py` 가 씀) |
| `train.py` | **학습** (끊겨도·곡을 더해도 이어서) |
| `generate.py` | 평가: 평가용 곡으로 악보를 만들어 사람 악보와 F1 비교 |
| `beat_eval.py`, `beat_grid.py` | 박 찾기(beat_this) 평가와 보정 |
| `bench.py`, `TIMINGS.md` | 단계별 시간 측정과 폰 예상 |

> ⚠️ 데이터 라이선스: osu-beatmaps 는 "연구 목적, 곡 저작권은 각 소유자"인 데이터. 시제품·연구용으로 쓰고, 상용 출시 전에 판단할 것.

---

## 1. 준비 (한 번)

필요한 것: **Python 3.10 이상**, **ffmpeg**, **curl** (윈도우 10 이상은 기본 포함), 디스크 **약 10GB** (데이터 4.3GB + 받는 중 임시 1GB), RAM **8GB 이상** 권장 (학습 때 데이터를 전부 메모리에 올림 — 약 5GB).

```bash
git clone -b taptaping-light https://github.com/jha21vvv/Mapperatorinator.git
cd Mapperatorinator
python -m venv .venv
```
- 윈도우: `.venv\Scripts\activate` / 맥·리눅스: `source .venv/bin/activate`

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```
- 박 찾기 평가까지 하려면 추가로: `pip install librosa beat-this`
- ffmpeg 확인: `ffmpeg -version` 이 나와야 함 (윈도우는 https://www.gyan.dev/ffmpeg/builds/ 에서 받아 PATH 에 추가)

## 2. 데이터 받기 (약 30분, 4K 곡 약 1,270곡)

```bash
python taptaping/light/extract.py --shards 3 37 0 7 14 20 27 33 40 47 53 60 67 73 --out data_light
```
- 조각(1GB)을 하나씩 받아 4K 곡만 뽑고 **바로 지운다** → 디스크는 결과물(약 4.3GB)만 남음.
- 끊겨도 같은 명령을 다시 실행하면 끝난 조각은 건너뛴다 (`data_light/extract_log.jsonl` 에 기록).
- 결과: `data_light/songs/*.npz` (곡마다 멜·박·악보), `data_light/audio/*.opus` (평가용 5% 곡의 오디오)
- 클라우드에서 같은 조각으로 뽑은 결과: 곡 1,273개 · 악보 3,503개 · 평가용 71곡. 같은 조각을 쓰면 평가용 곡도 같다 (곡 해시로 정함).

## 3. 학습 (하루~이틀)

```bash
python taptaping/light/train.py --data data_light --ckpt runs/light.pt --hours 36 --threads 0
```
- `--threads 0` = 모든 코어 사용. PC 를 같이 쓰려면 코어 수보다 2 정도 적게 (예: 8코어면 `--threads 6`).
- `--hours` 시간이 지나면 저장하고 끝난다. **20분마다 체크포인트 저장** (`runs/light.pt`).
- **끊겼을 때**: 같은 명령을 다시 실행하면 `runs/light.pt` 에서 이어서 학습.
- 처음 1~2분은 곡을 메모리에 올리느라 아무 출력이 없음 (정상).

로그 보는 법:
```
단계 11700  손실 1.357  57.0 샘플/초  lr 2.97e-04      ← 50단계마다 (손실이 내려가면 배우는 중)
== 평가: 손실 1.144, 다음 토큰 정확도 63.8%            ← 저장할 때마다, 평가용 곡 기준 (클라우드 약 2시간 학습 시점 실제 값)
```
- 참고 (클라우드 4코어): 초당 약 57 샘플. 집 CPU 코어가 많으면 더 빠름.
- 평가 손실이 몇 번 연속 오르면 과적합 신호 → 그때 멈춰도 됨 (마지막 저장본 사용).

## 4. 점진적으로 데이터 늘리기

```bash
python taptaping/light/extract.py --shards 1 10 17 24 30 36 43 50 56 63 70 --out data_light   # 조각 추가
python taptaping/light/train.py --data data_light --ckpt runs/light.pt --hours 24               # 같은 체크포인트에서 이어서
```
- 조각 번호는 0~73. 이미 받은 조각은 건너뛴다. 평가용 곡은 곡 해시로 정해져 학습에 섞이지 않는다.

## 5. 평가

```bash
python taptaping/light/generate.py --ckpt runs/light.pt --eval-data data_light --n 20
```
- 평가용 곡의 사람 악보와 비교: **시간 F1** (노트 위치, ±1/48박), **레인까지 F1**, 노트 수 비율, 악보당 생성 시간.

## 6. 결과 보내기

`runs/light.pt` (약 40MB — 모델 + 이어 학습용 정보) 를 대화창에 올려 주면, 평가·ONNX 변환·Unity 이식을 이어서 한다.
