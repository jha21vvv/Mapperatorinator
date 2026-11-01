"""python -m unittest discover -s taptaping/tests  (레포 루트에서)"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import osu_to_taptaping as conv  # noqa: E402


def make_osu(objects, mode=3, keys=4, version="Test", red=((0, 500.0),)):
    tp = "\n".join(f"{t},{bl},4,1,0,100,1,0" for t, bl in red)
    return f"""osu file format v14

[General]
AudioFilename: audio.mp3
Mode: {mode}

[Metadata]
Title:Song
TitleUnicode:노래
Artist:Artist
ArtistUnicode:가수
Creator:Mapperatorinator
Version:{version}

[Difficulty]
CircleSize:{keys}

[TimingPoints]
{tp}
1000,-50,4,1,0,100,0,0

[HitObjects]
""" + "\n".join(objects) + "\n"


# 4K 열 중심 x: 64, 192, 320, 448
SAMPLE = [
    "64,192,1000,1,0,0:0:0:0:",
    "192,192,1500,128,0,2500:0:0:0:0:",
    "320,192,2000,1,0,0:0:0:0:",
    "448,192,2000,1,0,0:0:0:0:",
    "511,192,3000,1,0,0:0:0:0:",   # 경계값 → 레인 4
    "0,192,3500,1,0,0:0:0:0:",     # 경계값 → 레인 1
]


class ParseTests(unittest.TestCase):
    def test_lanes_times_holds(self):
        c = conv.parse_osu(make_osu(SAMPLE))
        self.assertEqual(c.mode, 3)
        self.assertEqual([n["lane"] for n in c.notes], [1, 2, 3, 4, 4, 1])
        self.assertEqual(c.notes[0], {"time": 1.0, "lane": 1, "type": 0, "duration": 0.0, "isSlide": 0})
        self.assertEqual(c.notes[1], {"time": 1.5, "lane": 2, "type": 1, "duration": 1.0, "isSlide": 0})
        self.assertEqual(c.main_bpm(), 120.0)

    def test_main_bpm_picks_longest_section(self):
        c = conv.parse_osu(make_osu(SAMPLE, red=((0, 500.0), (1000, 400.0))))
        self.assertEqual(c.main_bpm(), 150.0)  # 1000~3500ms 구간이 더 김

    def test_overlap_warning(self):
        objs = ["64,192,1000,128,0,3000:0:0:0:0:", "64,192,2000,1,0,0:0:0:0:"]
        c = conv.parse_osu(make_osu(objs))
        self.assertTrue(any("겹치는" in w for w in c.warnings))
        self.assertEqual(len(c.notes), 2)

    def test_zero_length_hold_becomes_tap(self):
        c = conv.parse_osu(make_osu(["64,192,1000,128,0,1000:0:0:0:0:"]))
        self.assertEqual(c.notes[0]["type"], 0)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, objects, **kw):
        p = self.dir / name
        p.write_text(make_osu(objects, **kw), encoding="utf-8")
        return str(p)

    def test_rejects_non_mania_and_7k(self):
        with self.assertRaises(ValueError):
            conv.load_osu(self.write("std.osu", SAMPLE, mode=0))
        with self.assertRaises(ValueError):
            conv.load_osu(self.write("7k.osu", SAMPLE, keys=7))

    def test_auto_assign_by_density_and_cli(self):
        sparse = self.write("a.osu", SAMPLE[:2], version="A")
        dense = self.write("b.osu", SAMPLE, version="B")
        out = self.dir / "out.json"
        self.assertEqual(conv.main([dense, sparse, "-o", str(out)]), 0)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(data["easyNotes"]), 2)      # 밀도 낮은 쪽 → easy
        self.assertEqual(len(data["insaneNotes"]), 6)    # 높은 쪽 → insane
        self.assertEqual(data["normalNotes"], [])
        self.assertEqual(data["title"], "노래")
        self.assertEqual(data["globalOffset"], 0.0)
        self.assertEqual(data["bpm"], 120.0)
        self.assertIn("AI", data["description"])

    def test_fill_missing(self):
        one = self.write("a.osu", SAMPLE)
        out = self.dir / "out.json"
        self.assertEqual(conv.main(["--hard", one, "--fill-missing", "-o", str(out)]), 0)
        data = json.loads(out.read_text(encoding="utf-8"))
        for d in conv.DIFFICULTIES:
            self.assertEqual(len(data[f"{d}Notes"]), 6)


if __name__ == "__main__":
    unittest.main()
