#!/bin/bash
# Unity 없이 mono(mcs) 로 taptaping_gamemode 의 AI 악보 생성 C# 코드를 검증한다.
#   1) 컴파일 검사 (Unity/Inference Engine 대역으로)  2) 원본 기록 재연 (프롬프트·박자선·노트)  3) 토큰 확률 계산
# 준비: trace_generation.py → dump_csharp_vectors.py / dump_sampler_vectors.py, export_tables.py 결과
#   사용: run_tests.sh <taptaping_gamemode 경로> <vectors 폴더> <mapper_tables_unity.json>
set -e
GM=$1; VEC=$2; TABLES_JSON=$3
M=$GM/Assets/Base/Scripts/App/AIChart/Mapper
cd "$(dirname "$0")"
python3 -c "
import json,sys
t=json.load(open('$TABLES_JSON'))
with open('tables.txt','w') as f:
    for k,v in t.items():
        if k=='events':
            for e in v: f.write(f\"event {e['type']} {e['start']} {e['min']} {e['max']}\n\")
        else: f.write(f'{k} {v}\n')
"
mcs -langversion:latest -target:library -out:all.dll UnityStubs.cs UnityApiStubs.cs $M/*.cs
mcs -langversion:latest -out:replay.exe UnityStubs.cs $M/MapperTables.cs $M/MapperMel.cs $M/MapperTiming.cs $M/MapperGenerator.cs ReplayTest.cs
mcs -langversion:latest -out:sampler.exe UnityStubs.cs $M/MapperTables.cs $M/MapperMel.cs $M/MapperTiming.cs $M/MapperGenerator.cs SamplerTest.cs
LANG=C.UTF-8 mono replay.exe $VEC/full tables.txt
LANG=C.UTF-8 mono sampler.exe $VEC/sampler.txt tables.txt
