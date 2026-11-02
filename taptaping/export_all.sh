#!/bin/bash
# taptaping 기기용 AI 악보 모델 파일을 한 번에 만든다 (GPU 불필요, CPU 로 5분 안팎).
#   결과 폴더 구조 = Unity 메뉴 "TapTaping > AI Chart > 모델 변환" 에서 고를 폴더
#     <out>/mapper_tables.json, mapper_tables_unity.json
#     <out>/timing/{encoder.onnx, decoder.onnx, meta.json}   ← 박자 모델 (v32-mini 공용)
#     <out>/map/{encoder.onnx, decoder.onnx, meta.json}      ← 노트 모델 (v32-mini mania 전용)
#
# 사용 (레포 루트에서, requirements.txt 설치된 Python 3.10+):
#   bash taptaping/export_all.sh 출력폴더 아무오디오.mp3
#   (오디오는 설정 검증용으로만 쓰임 — 내용 무관)
set -e
OUT=${1:?출력 폴더}
AUDIO=${2:?아무 오디오 파일 (mp3/ogg)}
cd "$(dirname "$0")/.."
export PYTHONPATH=.
pip install -q onnx onnxruntime onnxscript
python taptaping/export_kv.py --subfolder "" --out "$OUT/timing"
python taptaping/export_kv.py --subfolder gamemode=3 --out "$OUT/map"
python taptaping/mini/export_tables.py --audio "$AUDIO" -o "$OUT/mapper_tables.json"
echo
echo "완료: $OUT"
echo "다음: Unity 에서 TapTaping > AI Chart > 모델 변환 (ONNX → fp16 .sentis) 메뉴로 이 폴더를 고르세요."
