namespace llmtest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Core core = new();
            core.Run().Wait();
        }
    }
}
