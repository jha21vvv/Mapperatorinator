"""생성에 필요한 토크나이저 표와 상수를 JSON 하나로 뽑는다 (Python 참조 구현과 C# 이 같이 읽음).

  PYTHONPATH=. python taptaping/mini/export_tables.py -o exports/mapper_tables.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--audio", required=True, help="설정 검증용 아무 오디오 파일 (내용은 안 씀)")
    a = ap.parse_args()
    import inference as inf
    from bench_speedups import make_args
    args = make_args([f"audio_path={a.audio}", "output_path=.", "gamemode=3", "keycount=4", "device=cpu", "difficulty=2.5"])
    _, tok = inf.load_model_with_server(args.model_path, args.train, "cpu", gamemode=3, auto_select_gamemode_model=True,
                                        precision="fp32", attn_implementation="sdpa")
    d = args.train.data
    out = {
        "offset": tok.offset,
        "pad_id": tok.pad_id, "sos_id": tok.sos_id, "eos_id": tok.eos_id,
        "event_start": {k.value: v for k, v in tok.event_start.items()},
        "event_range": {k.value: [r.min_value, r.max_value] for k, r in tok.event_range.items()},
        "event_order": [er.type.value for er in tok.event_ranges] + [er.type.value for er in tok.input_event_ranges],
        "context_sos": {k.value: v for k, v in tok.context_sos.items()},
        "context_eos": {k.value: v for k, v in tok.context_eos.items()},
        "vocab_size_out": tok.vocab_size_out, "vocab_size_in": tok.vocab_size_in,
        "num_diff_classes": tok.num_diff_classes, "max_difficulty": tok.max_difficulty,
        "num_mapper_classes": tok.num_mapper_classes, "num_descriptor_classes": tok.num_descriptor_classes,
        "sample_rate": d.sample_rate, "hop_length": d.hop_length, "src_seq_len": d.src_seq_len,
        "tgt_seq_len": d.tgt_seq_len, "train_lookahead": d.lookahead, "ms_per_step": 10,
        "temperature": args.temperature, "top_p": args.top_p, "timing_leniency": args.timing_leniency,
        "sustain_interval": d.sustain_interval,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(f"→ {a.output}")
    unity = to_unity(out)
    up = Path(a.output).with_name("mapper_tables_unity.json")
    up.write_text(json.dumps(unity, indent=1))
    print(f"→ {up} (Unity JsonUtility 용: 딕셔너리 대신 배열)")


def to_unity(t: dict) -> dict:
    """Unity JsonUtility 는 딕셔너리를 못 읽으므로 배열·필드로 펼친다 (MapperTables.cs 와 짝)."""
    return {
        "offset": t["offset"], "padId": t["pad_id"], "sosId": t["sos_id"], "eosId": t["eos_id"],
        "events": [{"type": typ, "start": t["event_start"][typ], "min": t["event_range"][typ][0],
                    "max": t["event_range"][typ][1]} for typ in t["event_order"]],
        "timingSos": t["context_sos"]["timing"], "timingEos": t["context_eos"]["timing"],
        "mapSos": t["context_sos"]["map"], "mapEos": t["context_eos"]["map"],
        "svSos": t["context_sos"].get("sv", -1), "svEos": t["context_eos"].get("sv", -1),
        "vocabOut": t["vocab_size_out"], "vocabIn": t["vocab_size_in"],
        "numDiffClasses": t["num_diff_classes"], "maxDifficulty": t["max_difficulty"],
        "numMapperClasses": t["num_mapper_classes"], "numDescriptorClasses": t["num_descriptor_classes"],
        "sampleRate": t["sample_rate"], "hopLength": t["hop_length"], "srcSeqLen": t["src_seq_len"],
        "tgtSeqLen": t["tgt_seq_len"], "timingLeniency": t["timing_leniency"],
        "temperature": t["temperature"], "topP": t["top_p"],
    }


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--unity-only":
    # 이미 뽑은 mapper_tables.json → mapper_tables_unity.json 만 다시 만들기
    src = Path(sys.argv[2])
    src.with_name("mapper_tables_unity.json").write_text(json.dumps(to_unity(json.loads(src.read_text())), indent=1))
    sys.exit(0)

if __name__ == "__main__":
    main()
