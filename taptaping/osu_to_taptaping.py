"""osu!mania 4K .osu 파일을 taptaping 악보(ChartExportData JSON)로 바꾼다.

taptaping 쪽 규격 (taptaping_gamemode):
  - NoteData        : Assets/Base/Scripts/Engine/Data/MusicScore.cs
      time(초), lane(1~4, 1이 맨 왼쪽), type(0 단타 / 1 롱노트), duration(초, 롱노트만), isSlide(0)
  - ChartExportData : Assets/Base/Scripts/Engine/Data/ChartExportData.cs
      easyNotes / normalNotes / hardNotes / insaneNotes 네 난이도 목록 + 곡 메타데이터

변환 규칙 (Mapperatorinator 의 postprocessor.py 와 같은 열 계산):
  - 레인     : lane = clamp(floor(x * 4 / 512), 0, 3) + 1
  - 시간     : time = 시작 ms / 1000   (globalOffset 은 0 — 노트 시간이 곧 음원 기준 시간)
  - 롱노트   : 타입 비트 128 → type 1, duration = (끝 ms − 시작 ms) / 1000
  - 그 외    : type 0, duration 0
  - BPM      : 곡에서 가장 오래 유지되는 빨간 선(uninherited timing point)의 BPM

사용 예:
  # 난이도를 직접 지정
  python taptaping/osu_to_taptaping.py --easy e.osu --normal n.osu --hard h.osu --insane i.osu -o chart.json

  # 파일만 나열하면 초당 노트 수(밀도) 순으로 쉬움→인세인 자동 배정
  python taptaping/osu_to_taptaping.py a.osu b.osu c.osu d.osu -o chart.json

표준 라이브러리만 쓴다 (설치 없이 실행 가능).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DIFFICULTIES = ("easy", "normal", "hard", "insane")
KEY_COUNT = 4
AI_DISCLOSURE = "AI(Mapperatorinator)로 생성한 악보입니다."


@dataclass
class OsuChart:
    path: str
    mode: int = 0
    key_count: float = 0.0
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""
    audio_filename: str = ""
    # (time_ms, beat_length) — 빨간 선만
    red_lines: list[tuple[float, float]] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def length_sec(self) -> float:
        if not self.notes:
            return 0.0
        first = self.notes[0]["time"]
        last = max(n["time"] + n["duration"] for n in self.notes)
        return max(last - first, 1e-6)

    @property
    def density(self) -> float:
        """초당 노트 수 (난이도 자동 배정용)"""
        return len(self.notes) / self.length_sec if self.notes else 0.0

    def main_bpm(self) -> float:
        if not self.red_lines:
            return 0.0
        end_ms = max((n["time"] + n["duration"]) * 1000 for n in self.notes) if self.notes else self.red_lines[-1][0]
        best_len, best_bpm = -1.0, 0.0
        lines = sorted(self.red_lines)
        for i, (t, beat_len) in enumerate(lines):
            nxt = lines[i + 1][0] if i + 1 < len(lines) else max(end_ms, t)
            span = nxt - t
            if beat_len > 0 and span > best_len:
                best_len, best_bpm = span, 60000.0 / beat_len
        return round(best_bpm, 3)


def parse_osu(text: str, path: str = "<text>") -> OsuChart:
    chart = OsuChart(path=path)
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue

        if section in ("General", "Metadata", "Difficulty"):
            if ":" not in line:
                continue
            key, value = (s.strip() for s in line.split(":", 1))
            if key == "Mode":
                chart.mode = int(value)
            elif key == "AudioFilename":
                chart.audio_filename = value
            elif key == "Title":
                chart.title = value
            elif key == "TitleUnicode":
                chart.title_unicode = value
            elif key == "Artist":
                chart.artist = value
            elif key == "ArtistUnicode":
                chart.artist_unicode = value
            elif key == "Creator":
                chart.creator = value
            elif key == "Version":
                chart.version = value
            elif key == "CircleSize":
                chart.key_count = float(value)

        elif section == "TimingPoints":
            parts = line.split(",")
            if len(parts) < 2:
                continue
            t, beat_len = float(parts[0]), float(parts[1])
            uninherited = int(parts[6]) if len(parts) > 6 else (1 if beat_len > 0 else 0)
            if uninherited == 1 and beat_len > 0:
                chart.red_lines.append((t, beat_len))

        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 5:
                chart.warnings.append(f"읽을 수 없는 노트 줄: {line}")
                continue
            x = float(parts[0])
            t_ms = float(parts[2])
            obj_type = int(parts[3])
            lane = min(max(int(math.floor(x * KEY_COUNT / 512)), 0), KEY_COUNT - 1) + 1

            if obj_type & 128:
                end_ms = float(parts[5].split(":", 1)[0]) if len(parts) > 5 else t_ms
                if end_ms <= t_ms:
                    chart.warnings.append(f"길이가 0 이하인 롱노트를 단타로 처리: {int(t_ms)}ms 레인 {lane}")
                    chart.notes.append(_note(t_ms, lane, 0, 0.0))
                else:
                    chart.notes.append(_note(t_ms, lane, 1, end_ms - t_ms))
            else:
                chart.notes.append(_note(t_ms, lane, 0, 0.0))

    chart.notes.sort(key=lambda n: (n["time"], n["lane"]))
    _check_overlaps(chart)
    return chart


def _note(t_ms: float, lane: int, note_type: int, dur_ms: float) -> dict:
    # 키 순서는 NoteData 필드 순서와 같게
    return {
        "time": round(t_ms / 1000.0, 4),
        "lane": lane,
        "type": note_type,
        "duration": round(dur_ms / 1000.0, 4),
        "isSlide": 0,
    }


def _check_overlaps(chart: OsuChart) -> None:
    """같은 레인에서 노트가 롱노트 안에 겹치면 경고만 남긴다 (데이터는 바꾸지 않음)."""
    busy_until = {lane: -1.0 for lane in range(1, KEY_COUNT + 1)}
    overlaps = 0
    for n in chart.notes:
        if n["time"] < busy_until[n["lane"]] - 1e-4:
            overlaps += 1
        busy_until[n["lane"]] = max(busy_until[n["lane"]], n["time"] + n["duration"])
    if overlaps:
        chart.warnings.append(f"같은 레인에서 겹치는 노트 {overlaps}개 (그대로 둠)")


def load_osu(path: str | Path) -> OsuChart:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    chart = parse_osu(text, str(path))
    if chart.mode != 3:
        raise ValueError(f"{path}: osu!mania 악보가 아닙니다 (Mode={chart.mode}, 필요: 3)")
    if round(chart.key_count) != KEY_COUNT:
        raise ValueError(f"{path}: 4키 악보가 아닙니다 (키 수 {chart.key_count:g})")
    if not chart.notes:
        raise ValueError(f"{path}: 노트가 없습니다")
    return chart


def assign_difficulties(explicit: dict[str, str | None], loose: list[str]) -> dict[str, OsuChart | None]:
    """명시한 난이도는 그대로, 나머지 파일은 밀도 순으로 빈 난이도 칸에 채운다."""
    result: dict[str, OsuChart | None] = {d: None for d in DIFFICULTIES}
    for d, p in explicit.items():
        if p:
            result[d] = load_osu(p)

    loose_charts = sorted((load_osu(p) for p in loose), key=lambda c: c.density)
    empty = [d for d in DIFFICULTIES if result[d] is None]
    if len(loose_charts) > len(empty):
        raise ValueError(f"난이도 칸({len(empty)}개)보다 파일({len(loose_charts)}개)이 많습니다")
    # 파일이 칸보다 적으면 칸 전체에 고르게 퍼뜨린다 (예: 2개 → easy, hard 가 아니라 easy, insane 쪽으로)
    if loose_charts:
        if len(loose_charts) == 1:
            slots = [empty[len(empty) // 2]] if len(empty) > 1 else empty[:1]
        else:
            step = (len(empty) - 1) / (len(loose_charts) - 1)
            slots = [empty[round(i * step)] for i in range(len(loose_charts))]
        for slot, chart in zip(slots, loose_charts):
            result[slot] = chart
    return result


def fill_missing(charts: dict[str, OsuChart | None]) -> list[str]:
    """빈 난이도를 가장 가까운 난이도의 노트로 채운다. 채운 내역을 돌려준다."""
    filled = []
    have = [i for i, d in enumerate(DIFFICULTIES) if charts[d] is not None]
    for i, d in enumerate(DIFFICULTIES):
        if charts[d] is None and have:
            j = min(have, key=lambda k: (abs(k - i), k))  # 같으면 쉬운 쪽
            charts[d] = charts[DIFFICULTIES[j]]
            filled.append(f"{d} ← {DIFFICULTIES[j]}")
    return filled


def build_export(charts: dict[str, OsuChart | None], title: str | None = None, artist: str | None = None,
                 creator: str | None = None, audio_file: str | None = None) -> dict:
    present = [c for c in charts.values() if c is not None]
    if not present:
        raise ValueError("변환할 악보가 없습니다")
    ref = present[0]
    bpms = [c.main_bpm() for c in present if c.main_bpm() > 0]

    return {
        "chartId": str(uuid.uuid4()),
        "title": title or ref.title_unicode or ref.title,
        "artist": artist or ref.artist_unicode or ref.artist,
        "creator": creator or ref.creator,
        "description": AI_DISCLOSURE,
        "genreTag": "",
        "genreEdited": False,
        "groupName": "first",
        "bpm": bpms[0] if bpms else 0.0,
        "globalOffset": 0.0,
        "audioFileName": audio_file or ref.audio_filename,
        "coverImageBase64": "",
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "serverSongId": "",
        "serverChartVersion": 0,
        "sourceOption": 0,
        "verificationCode": "",
        "sourceUrl": "",
        **{f"{d}Notes": (charts[d].notes if charts[d] else []) for d in DIFFICULTIES},
    }


def summary(charts: dict[str, OsuChart | None]) -> str:
    lines = []
    for d in DIFFICULTIES:
        c = charts[d]
        if c is None:
            lines.append(f"  {d:<7} (비어 있음)")
            continue
        holds = sum(1 for n in c.notes if n["type"] == 1)
        per_lane = [sum(1 for n in c.notes if n["lane"] == k) for k in range(1, KEY_COUNT + 1)]
        lines.append(f"  {d:<7} {Path(c.path).name} [{c.version}]  노트 {len(c.notes)}개 (롱노트 {holds}), "
                     f"초당 {c.density:.2f}개, 레인별 {per_lane}, BPM {c.main_bpm():g}")
        for w in c.warnings:
            lines.append(f"          ⚠ {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="osu!mania 4K .osu → taptaping ChartExportData JSON")
    ap.add_argument("files", nargs="*", help="난이도를 자동 배정할 .osu 파일들 (초당 노트 수 순)")
    for d in DIFFICULTIES:
        ap.add_argument(f"--{d}", help=f"{d} 난이도로 쓸 .osu 파일")
    ap.add_argument("-o", "--output", required=True, help="저장할 JSON 경로")
    ap.add_argument("--fill-missing", action="store_true", help="빈 난이도를 가장 가까운 난이도로 채움")
    ap.add_argument("--title")
    ap.add_argument("--artist")
    ap.add_argument("--creator")
    ap.add_argument("--audio-file", help="audioFileName 값 (기본: .osu 의 AudioFilename)")
    args = ap.parse_args(argv)

    explicit = {d: getattr(args, d) for d in DIFFICULTIES}
    if not args.files and not any(explicit.values()):
        ap.error(".osu 파일을 하나 이상 주세요")

    try:
        charts = assign_difficulties(explicit, args.files)
        filled = fill_missing(charts) if args.fill_missing else []
        data = build_export(charts, args.title, args.artist, args.creator, args.audio_file)
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")

    print(f"저장: {args.output}")
    print(summary(charts))
    for f in filled:
        print(f"  채움: {f}")
    empty = [d for d in DIFFICULTIES if not data[f"{d}Notes"]]
    if empty:
        print(f"  ⚠ 빈 난이도: {', '.join(empty)} (게임에서 이 난이도를 고르면 다른 악보로 넘어감 — --fill-missing 참고)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
