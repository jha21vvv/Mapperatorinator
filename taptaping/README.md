# taptaping 4레인 AI 악보 — 작업 폴더

Mapperatorinator 를 taptaping(4레인 리듬게임)용으로 쓰기 위한 도구와 실험 기록.
게임 쪽 규격: `taptaping_gamemode` 의 `NoteData`(MusicScore.cs), `ChartExportData`(ChartExportData.cs).

## 파일

| 파일 | 하는 일 |
|---|---|
| `osu_to_taptaping.py` | 4K .osu → taptaping `ChartExportData` JSON (표준 라이브러리만 사용) |
| `tests/` | 변환기 단위 테스트 — `python -m unittest discover -s taptaping/tests` |
| `export_onnx.py` | 모델을 encoder/decoder ONNX 로 내보내고 int8 양자화·PyTorch 대조 (`PYTHONPATH=.` 필요) |

Unity 쪽 짝: `taptaping_gamemode/Assets/Base/Scripts/App/AIChart/OsuManiaImporter.cs`
(+ 에디터 창 `TapTaping > AI Chart > .osu 4K 가져오기`). Python 판과 같은 입력에 같은 결과가 나오는 것을 확인함.

## 4K 악보 생성 (원본 모델)

```bash
python inference.py -cn v32-mini audio_path=song.mp3 output_path=out gamemode=3 keycount=4 difficulty=3.0 hold_note_ratio=0.15
python taptaping/osu_to_taptaping.py out/*.osu -o chart.json --fill-missing
```

## 실험 기록 (2026-11-01, CPU 4코어 클라우드)

### 모델 구조 — v32-mini `gamemode=3` (mania 전용 체크포인트)
- 파라미터 5,580만 개, fp32 223MB. Whisper-base 구조(인코더 6층 + 디코더 6층, d=512), RoPE.
- 입력 어휘 12,505개 중 7,605개가 매퍼 ID 토큰 → 입력 임베딩(640만 파라미터)의 대부분. 4K 전용이면 잘라낼 수 있음.
- 입력: 16kHz, n_fft 1024, hop 128, 멜 128개(20~8000Hz, HTK), log1p. 창 하나 = 멜 2048프레임(약 16.4초) → 인코더 출력 1024.
- 주의: 생성할 때 **박자(timing)는 기본 모델(전 모드 공용)이, 노트는 mania 모델이** 만든다 → 모델이 2개 필요.

### 속도 (PyTorch, CPU 4코어)
- 40초 구간 생성: 약 30초 (모델 로딩 제외).

### ONNX 내보내기 결과 (`export_onnx.py`)
| 파일 | 크기 | 정확도 (PyTorch 대비) |
|---|---|---|
| encoder fp32 | 79.7MB | 오차 2e-6 |
| decoder fp32 (KV 캐시 없음) | 135.1MB | 다음 토큰 100% 일치 |
| encoder int8 (ORT 동적 양자화) | 20.2MB | 오차 0.13 |
| decoder int8 (ORT 동적 양자화) | 34.2MB | **다음 토큰 31% 일치 — 못 씀** |

- 내보내려면 `nn.RMSNorm` 을 기본 연산으로 풀어야 함 (opset 17 미지원) — 스크립트가 자동으로 교체, 오차 1e-7.
- `attn_implementation="eager"` 는 이 모델에서 깨져 있음 (마스크 None) → `sdpa` 로 내보냄.

### 가중치 양자화 시뮬레이션 (계산은 float, 가중치만 줄임 — Sentis 방식)
| 방식 | 다음 토큰 일치 |
|---|---|
| fp16 | 100% |
| uint8 텐서 단위 | 7.8% |
| uint8 행 단위 | 87.5% |

- 망가지는 주범은 **입력 임베딩(decoder_embedder)** — 이것만 양자화해도 4.7%. 그 외 층은 70~94%.
- 결론: 지금 모델 그대로면 **fp16(약 110MB/모델)이 안전선**. 더 줄이려면 임베딩은 fp16 유지 + 매퍼 토큰 제거 + 나머지 행 단위 int8, 또는 작은 모델 학습(증류) 필요.
- 위 수치는 무작위 입력 기준. 실제 곡으로 생성 품질 비교는 다음 단계.

### 속도 개선 실측 (2026-11-02, Track01 3분 19초, 4개 난이도, 시드 3개 평균)

`bench_speedups.py` + `compare_charts.py --ref-beats Track01.beats.json --ref-shift-ms 50`

| 설정 | 박자 창 | 노트 창 | 4개 난이도 합계 | 노트 1,000개당 |
|---|---|---|---|---|
| 기준 (inference.py 를 난이도마다) | 0.5/0.4 | 0.5/0.4 | 834초 | 234초 |
| ①② (SV 생략 + 모델·박자·인코더 공유) | 0.5/0.4 | 0.5/0.4 | 520초 (−38%) | 137초 |
| **혼합 ③ (노트만 창 겹침 축소)** | 0.5/0.4 | 0.25/0.25 | **396초 (−53%)** | 111초 |
| 전체 ③ | 0.25/0.25 | 0.25/0.25 | 386초 (−54%) | 106초 |

- 품질: 혼합 ③ 은 ①② 와 시드 편차 안에서 같음 (음악에 맞는 노트 91~94%). 박자는 ①② 와 동일.
- 전체 ③ 은 박자가 나빠짐: 엉뚱한 BPM 박자선 평균 7.3개 (①② 1.3개), 박자선 37개 (19개). 4.5 난이도 음악에 맞는 노트 86% (최저 77%).
- librosa 박 추적은 박을 약 50ms 늦게 잡음 → `--ref-shift-ms 50` 으로 보정.
- 모든 설정(기준 포함)에서 10~30초 빈 구간이 가끔 생김 — 주로 템포가 흔들리는 20~55초, 150~175초 부근. 속도 개선과 무관한 모델 특성.

## 기기용 생성 (Unity) — 2026-11-02

모델 파일 만들기: `bash taptaping/export_all.sh 출력폴더 아무오디오.mp3` → Unity 메뉴 `TapTaping > AI Chart > 모델 변환` 으로 fp16 .sentis.

| 파일 | 하는 일 |
|---|---|
| `export_kv.py` | KV 캐시 디코더 + 교차 K/V 를 미리 계산하는 인코더 ONNX (PyTorch 대비 로짓 오차 < 1e-3) |
| `mini/mapper_mini.py` | torch 없는 numpy+ONNX 참조 구현 (C# 이식 설계도) |
| `mini/verify_mini.py` | 원본 기록 재연 검증 (프롬프트·박 이벤트·박자선·노트 전부 일치) |
| `csharp_test/` | Unity C# 코드를 mono 로 같은 방식 검증 + 토큰 확률 비교 |

### 실측 (클라우드 CPU 4코어, ONNX Runtime, Track01 3분 19초)
- 토큰당: 노트 모델 3.6ms, 박자 모델 5.0ms (PyTorch generate 약 19ms 의 1/5)
- 4개 난이도 전체: **131초** (박자 30초 + 노트 101초) — PyTorch 혼합 ③ 396초의 1/3
- 품질: 노트 지표는 PyTorch 판과 시드 편차 안에서 같음. 박자선 수는 시드 6개 비교에서 ONNX 20~31개 / 원본 20~31개로 같은 분포.
- 다음 토큰 확률분포: 원본 HF 처리기와 차이 < 1e-6 (Python), < 3e-6 (C#)
