// 원본 기록을 그대로 돌려주는 가짜 모델로 C# 생성 루프 전체를 검증
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TapTaping.AIChart.Mapper;

class ReplayModel : IMapperModel
{
    public readonly Queue<(int[] prompt, int[] gen, string stage, double ft)> Windows = new Queue<(int[], int[], string, double)>();
    public int Vocab; public int PromptMismatch, WindowsUsed; public string Name;
    (int[] prompt, int[] gen, string stage, double ft) cur; int pos;
    public float[] FirstMel;
    public void Encode(float[] mel) { if (FirstMel == null) FirstMel = mel; }
    public float[] Start(int[] prompt)
    {
        cur = Windows.Dequeue(); WindowsUsed++; pos = 0;
        if (!prompt.SequenceEqual(cur.prompt))
        {
            PromptMismatch++;
            if (PromptMismatch <= 2)
            {
                int d = Enumerable.Range(0, Math.Min(prompt.Length, cur.prompt.Length)).FirstOrDefault(i => prompt[i] != cur.prompt[i]);
                Console.WriteLine($"  [{Name}] 프롬프트 다름 frame {cur.ft}: 위치 {d}, 길이 {prompt.Length}/{cur.prompt.Length}");
            }
        }
        return Forced();
    }
    public float[] Step(int token) => Forced();
    float[] Forced()
    {
        var l = new float[Vocab];
        for (int i = 0; i < Vocab; i++) l[i] = -1e9f;
        if (pos < cur.gen.Length) l[cur.gen[pos]] = 0f;
        pos++;
        return l;
    }
    public void Dispose() { }
}

static class Program
{
    static int[] Ints(string s) { return s.Trim().Length == 0 ? new int[0] : s.Trim().Split(' ').Select(int.Parse).ToArray(); }
    static bool ok = true;
    static MapperTables LoadTablesText(string path)
    {
        var t = new MapperTables();
        var evs = new List<MapperEventRange>();
        foreach (var line in File.ReadAllLines(path))
        {
            var p = line.Split(' ');
            if (p[0] == "event") { evs.Add(new MapperEventRange { type = p[1], start = int.Parse(p[2]), min = int.Parse(p[3]), max = int.Parse(p[4]) }); continue; }
            var f = typeof(MapperTables).GetField(p[0]);
            if (f == null) continue;
            f.SetValue(t, f.FieldType == typeof(int) ? (object)int.Parse(p[1]) : float.Parse(p[1], CultureInfo.InvariantCulture));
        }
        t.events = evs.ToArray();
        t.Init();
        return t;
    }
    static void Check(string n, bool c, string d) { Console.WriteLine("[" + (c ? "OK" : "틀림") + "] " + n + " " + d); ok &= c; }
    static int Main(string[] args)
    {
        string dir = args[0], tablesPath = args[1];
        var k = LoadTablesText(tablesPath);
        var bytes = File.ReadAllBytes(Path.Combine(dir, "samples.f32"));
        var samples = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);
        var st = File.ReadAllText(Path.Combine(dir, "settings.txt")).Trim().Split(' ');
        float diff = float.Parse(st[0], CultureInfo.InvariantCulture);
        var tw = st[1].Split(',').Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        var mw = st[2].Split(',').Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();

        var timing = new ReplayModel { Vocab = k.vocabOut, Name = "박자" };
        var map = new ReplayModel { Vocab = k.vocabOut, Name = "노트" };
        var lines = File.ReadAllLines(Path.Combine(dir, "windows.txt"));
        for (int i = 0; i + 2 < lines.Length + 1; i += 3)
        {
            var h = lines[i].Split(' ');
            var w = (Ints(lines[i + 1]), Ints(lines[i + 2]), h[0], double.Parse(h[1], CultureInfo.InvariantCulture));
            (h[0] == "timing" ? timing : map).Windows.Enqueue(w);
        }
        int nT = timing.Windows.Count, nM = map.Windows.Count;
        var s = new MapperSettings { timingDifficulty = diff, timingLookback = tw[0], timingLookahead = tw[1], mapLookback = mw[0], mapLookahead = mw[1] };
        var g = new MapperGenerator(k, timing, map, s);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var e = g.Run(samples, new[] { diff });
        while (e.MoveNext()) { }
        Console.WriteLine($"실행 {sw.Elapsed.TotalSeconds:F1}초 (멜 계산 포함, 모델 없음)");
        Check("박자 창 수", timing.WindowsUsed == nT, $"{timing.WindowsUsed}/{nT}");
        Check("박자 프롬프트", timing.PromptMismatch == 0, $"다른 창 {timing.PromptMismatch}");
        Check("노트 창 수", map.WindowsUsed == nM, $"{map.WindowsUsed}/{nM}");
        Check("노트 프롬프트", map.PromptMismatch == 0, $"다른 창 {map.PromptMismatch}");

        var expTp = File.ReadAllLines(Path.Combine(dir, "timing_points.txt")).Where(x => x.Trim().Length > 0)
            .Select(x => x.Split(' ').Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToList();
        var tp = g.Result.Timing;
        bool tpOk = tp.Count == expTp.Count && tp.Zip(expTp, (a, b) => Math.Abs(a.Offset - b[0]) < 1e-6 && Math.Abs(a.MsPerBeat - b[1]) < 1e-6 && a.Meter == (int)b[2]).All(x => x);
        Check("박자선", tpOk, $"{tp.Count}/{expTp.Count}");

        var exp = File.ReadAllLines(Path.Combine(dir, "expected_notes.txt")).Where(x => x.Trim().Length > 0)
            .Select(x => x.Split(' ').Select(int.Parse).ToArray()).Select(a => (a[0], a[1], a[2])).OrderBy(x => x).ToList();
        var got = g.Result.Notes[diff].Select(n => (n.StartMs, n.Lane, n.EndMs)).OrderBy(x => x).ToList();
        bool notesOk = exp.SequenceEqual(got);
        Check("노트", notesOk, $"{got.Count}/{exp.Count}");
        if (!notesOk) for (int i = 0, shown = 0; i < Math.Min(exp.Count, got.Count) && shown < 5; i++) if (!exp[i].Equals(got[i])) { Console.WriteLine("   원본 " + exp[i] + " / C# " + got[i]); shown++; }

        var melBytes = File.ReadAllBytes(Path.Combine(dir, "mel0.f32"));
        var melRef = new float[melBytes.Length / 4];
        Buffer.BlockCopy(melBytes, 0, melRef, 0, melBytes.Length);
        double maxRef = melRef.Max(Math.Abs), maxErr = 0;
        for (int i = 0; i < melRef.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(melRef[i] - map.FirstMel[i]));
        Check("멜 (첫 노트 창)", maxErr / maxRef < 1e-4, $"상대 오차 {maxErr / maxRef:E2}");
        Console.WriteLine(ok ? "\n전체: 통과" : "\n전체: 실패 있음");
        return ok ? 0 : 1;
    }
}
