// C# Process(로짓 처리) 결과 확률을 원본(HF 처리기) 확률과 비교
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TapTaping.AIChart.Mapper;

static class SamplerTest
{
    static int Main(string[] a)
    {
        var k = TablesText.Load(a[1]);
        var gen = new MapperGenerator(k, null, null, new MapperSettings());
        var lines = File.ReadAllLines(a[0]);
        double worst = 0; int n = 0;
        for (int i = 0; i + 3 < lines.Length + 1; i += 4)
        {
            int lbEnd = int.Parse(lines[i]);
            var seq = lines[i + 1].Split(' ').Select(int.Parse).ToList();
            var logits = lines[i + 2].Split(' ').Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            var refP = new double[logits.Length];
            foreach (var kv in lines[i + 3].Split(' ')) { var p = kv.Split(':'); refP[int.Parse(p[0])] = double.Parse(p[1], CultureInfo.InvariantCulture); }
            var x = gen.Process(logits, seq, lbEnd);
            double max = x.Max(), z = x.Sum(v => double.IsNegativeInfinity(v) ? 0 : Math.Exp(v - max));
            double d = 0;
            for (int j = 0; j < x.Length; j++) d = Math.Max(d, Math.Abs((double.IsNegativeInfinity(x[j]) ? 0 : Math.Exp(x[j] - max) / z) - refP[j]));
            worst = Math.Max(worst, d); n++;
        }
        Console.WriteLine($"케이스 {n}개, 원본 대비 최대 확률 차이 {worst:E2}");
        Console.WriteLine(worst < 1e-4 ? "통과" : "실패");
        return worst < 1e-4 ? 0 : 1;
    }
}
static class TablesText
{
    public static MapperTables Load(string path)
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
        t.events = evs.ToArray(); t.Init(); return t;
    }
}
