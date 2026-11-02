// Unity 없이 mono 로 컴파일하기 위한 최소 대역
namespace UnityEngine
{
    public static class JsonUtility
    {
        public static T FromJson<T>(string json) { throw new System.NotSupportedException("테스트에서는 LoadTablesText 사용"); }
        public static string ToJson(object o) => "";
    }
}
