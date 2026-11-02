// 컴파일 검사용 Unity / Inference Engine 대역 (실제 API 와 이름·시그니처만 맞춤 — 동작 없음)
using System;
namespace UnityEngine
{
    public class Object { public string name; }
    public class Component : Object { public GameObject gameObject; public T GetComponent<T>() where T : Component => null; }
    public class Behaviour : Component { }
    public class MonoBehaviour : Behaviour { public Coroutine StartCoroutine(System.Collections.IEnumerator e) => null; }
    public class Coroutine { }
    public class GameObject : Object { public GameObject(string n) { } public T AddComponent<T>() where T : Component => null; public T GetComponent<T>() where T : Component => null; }
    public enum AudioDataLoadState { Unloaded, Loading, Loaded, Failed }
    public class AudioClip : Object { public int channels, frequency, samples; public float length; public AudioDataLoadState loadState; public bool LoadAudioData() => true; public bool GetData(float[] d, int o) => true; }
    public static class Debug { public static void Log(object o) { } public static void LogWarning(object o) { } public static void LogError(object o) { } }
    public static class Time { public static double realtimeSinceStartupAsDouble; public static float realtimeSinceStartup; }
    public static class SystemInfo { public static string deviceModel, graphicsDeviceName, processorType; public static int processorCount, systemMemorySize; public static bool supportsComputeShaders; }
    public static class PlayerPrefs { public static string GetString(string k, string d) => d; public static void SetString(string k, string v) { } public static void Save() { } public static void DeleteKey(string k) { } }
    public static class Application { public static string persistentDataPath = "", dataPath = ""; }
    public static class Resources { public static T Load<T>(string p) where T : Object => null; }
    public class TooltipAttribute : Attribute { public TooltipAttribute(string s) { } }
    public static class Screen { public static float dpi; public static int width, height; }
    public struct Vector2 { public float x, y; }
    public struct Vector3 { public Vector3(float x, float y, float z) { } }
    public struct Rect { public Rect(float a, float b, float c, float d) { } }
    public struct Matrix4x4 { public static Matrix4x4 Scale(Vector3 v) => default; }
    public static class Mathf { public static float Max(float a, float b) => Math.Max(a, b); }
    public static class GUI { public static Matrix4x4 matrix; public static bool enabled; }
    public class GUILayoutOption { }
    public static class GUILayout
    {
        public static void BeginArea(Rect r) { } public static void EndArea() { } public static void Label(string s) { }
        public static bool Button(string s, params GUILayoutOption[] o) => false; public static GUILayoutOption Height(float h) => null;
        public static void BeginHorizontal() { } public static void EndHorizontal() { }
        public static Vector2 BeginScrollView(Vector2 v) => v; public static void EndScrollView() { }
    }
    public static class JsonUtility2 { }
}
namespace UnityEngine.Networking
{
    public class DownloadHandler { }
    public class DownloadHandlerFile : DownloadHandler { public DownloadHandlerFile(string p) { } public bool removeFileOnAbort; }
    public class UnityWebRequestAsyncOperation { public bool isDone; }
    public class UnityWebRequest : IDisposable
    {
        public enum Result { InProgress, Success, ConnectionError, ProtocolError, DataProcessingError }
        public static UnityWebRequest Get(string u) => null; public DownloadHandler downloadHandler; public float downloadProgress; public Result result; public string error;
        public UnityWebRequestAsyncOperation SendWebRequest() => null; public void Dispose() { }
    }
}
namespace TapTaping.Network { public class ApiConfig { public static ApiConfig Current; public string BaseUrl; } }
namespace Unity.InferenceEngine
{
    public enum BackendType { GPUCompute, CPU, GPUPixel }
    public class Model { }
    public class ModelAsset : UnityEngine.Object { }
    public static class ModelLoader { public static Model Load(string p) => null; public static Model Load(ModelAsset a) => null; }
    public struct TensorShape { public TensorShape(params int[] d) { } }
    public abstract class Tensor : IDisposable { public void Dispose() { } }
    public class Tensor<T> : Tensor where T : struct
    {
        public Tensor(TensorShape s) { } public Tensor(TensorShape s, T[] d) { }
        public Tensor<T> ReadbackAndClone() => this; public T[] DownloadToArray() => null;
    }
    public class Worker : IDisposable
    {
        public Worker(Model m, BackendType b) { }
        public void Schedule(params Tensor[] inputs) { } public void SetInput(string n, Tensor t) { }
        public Tensor PeekOutput(string n) => null; public void CopyOutput(string n, ref Tensor t) { }
        public void Dispose() { }
    }
}
// 게임 쪽 형식 (실제 정의는 Engine/Data)
public class NoteData { public float time; public int lane; public int type; public float duration; public int isSlide; }
public class ChartExportData
{
    public string chartId, title, artist, creator, description, genreTag, groupName, audioFileName, coverImageBase64, createdAt;
    public bool genreEdited; public float bpm, globalOffset; public int sourceOption;
    public System.Collections.Generic.List<NoteData> easyNotes, normalNotes, hardNotes, insaneNotes;
}
public class LocalFileChartUploader { public void UploadChart(ChartExportData d, Action<bool, string> cb) { } }
