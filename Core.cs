using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LLama;
using LLama.Common;

namespace llmtest
{
    public class Core
    {
        public async Task Run()
        {
            var prompt = File.ReadAllText("./prompt.txt");
            const string InstructionPrefix = "[INST]";
            const string InstructionSuffix = "[/INST]";

            string modelPath = "./models/phi-2.Q5_K_M.gguf"; // change it to your own model path

            // Load a model
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = 32768,
                Seed = 1337,
                SplitMode = LLama.Native.GPUSplitMode.Layer,
                GpuLayerCount = 5
            };
            using var model = LLamaWeights.LoadFromFile(parameters);
            

            // Initialize a chat session
            using var context = model.CreateContext(parameters);
            
            var ex = new InstructExecutor(context, InstructionPrefix, InstructionSuffix, null);

            // show the prompt
            //Console.WriteLine();
            //Console.Write(prompt);

            var inferenceParams = new InferenceParams()
            {
                Temperature = 0.5f,
                MaxTokens = -1,
            };

            StringBuilder stb = new StringBuilder();
            stb.AppendLine(prompt);
            stb.AppendLine();
            stb.AppendLine();
            
            var question = stb.ToString() + "The topic is computers.";
            // run the inference in a loop to chat with LLM
            while (prompt != "stop")
            {
                await foreach (var text in ex.InferAsync(question, inferenceParams)) 
                {
                    Console.Write(text);
                }
                Console.WriteLine();
                question = stb.ToString() + Console.ReadLine() ?? "";
            }

            // save the session
            //session.SaveSession("./Result");
        }
    }
}
