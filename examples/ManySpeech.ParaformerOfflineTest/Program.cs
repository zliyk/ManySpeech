using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
using PreProcessUtils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ManySpeech.ParaformerOfflineTest;

internal static class Program
{
    private static readonly string[] SupportedModels = new[]
    {
        "paraformer",
        "sensevoicesmall",
        "seacoparaformer"
    };

    public static int Main(string[] args)
    {
        try
        {
            var options = RecognitionOptions.Parse(args);
            ValidateOptions(options);
            var fileSet = ModelFileResolver.Resolve(options);
            ReportConfiguration(options, fileSet);
            RunRecognition(options, fileSet);
            return 0;
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"参数错误：{ex.Message}");
            RecognitionOptions.PrintUsage();
            return 2;
        }
        catch (FileNotFoundException ex)
        {
            Console.Error.WriteLine($"文件缺失：{ex.Message}");
            return 3;
        }
        catch (DirectoryNotFoundException ex)
        {
            Console.Error.WriteLine($"目录不存在：{ex.Message}");
            return 4;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("识别失败：");
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static void ValidateOptions(RecognitionOptions options)
    {
        if (!SupportedModels.Contains(options.ModelType, StringComparer.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"暂不支持的模型类型：{options.ModelType}，可选值：{string.Join(", ", SupportedModels)}");
        }
        if (!Directory.Exists(options.ModelDirectory))
        {
            throw new DirectoryNotFoundException(options.ModelDirectory);
        }
        if (!File.Exists(options.AudioPath))
        {
            throw new FileNotFoundException("音频文件不存在", options.AudioPath);
        }
    }

    private static void ReportConfiguration(RecognitionOptions options, ModelFileSet fileSet)
    {
        Console.WriteLine("=== 识别配置 ===");
        Console.WriteLine($"模型类型：{options.ModelType}");
        Console.WriteLine($"模型目录：{options.ModelDirectory}");
        Console.WriteLine($"模型文件：{fileSet.ModelFile}");
        Console.WriteLine($"配置文件：{fileSet.ConfigFile}");
        Console.WriteLine($"MVN 文件：{fileSet.MvnFile}");
        Console.WriteLine($"词表文件：{fileSet.TokensFile}");
        if (!string.IsNullOrEmpty(fileSet.ModelebFile))
        {
            Console.WriteLine($"Embed 文件：{fileSet.ModelebFile}");
        }
        if (!string.IsNullOrEmpty(fileSet.HotwordFile))
        {
            Console.WriteLine($"热词文件：{fileSet.HotwordFile}");
        }
        Console.WriteLine($"音频文件：{options.AudioPath}");
        Console.WriteLine($"线程数：{options.Threads}");
        Console.WriteLine();
    }

    private static void RunRecognition(RecognitionOptions options, ModelFileSet fileSet)
    {
        TimeSpan audioDuration = TimeSpan.Zero;
        float[] samples = AudioHelper.GetFileSample(options.AudioPath, ref audioDuration);
        if (samples.Length == 0 || audioDuration.TotalMilliseconds <= 0)
        {
            throw new InvalidOperationException("音频文件为空或无法确定时长");
        }

        using var recognizer = new OfflineRecognizer(
            modelFilePath: fileSet.ModelFile,
            configFilePath: fileSet.ConfigFile,
            mvnFilePath: fileSet.MvnFile,
            tokensFilePath: fileSet.TokensFile,
            modelebFilePath: fileSet.ModelebFile ?? string.Empty,
            hotwordFilePath: fileSet.HotwordFile ?? string.Empty,
            batchSize: 1,
            threadsNum: options.Threads);

        using var stream = recognizer.CreateOfflineStream();

        var watch = Stopwatch.StartNew();
        stream.AddSamples(samples);
        OfflineRecognizerResultEntity result = recognizer.GetResult(stream);
        watch.Stop();

        PrintResult(result);
        PrintPerformance(watch.Elapsed, audioDuration);
    }

    private static void PrintResult(OfflineRecognizerResultEntity result)
    {
        Console.WriteLine("=== 识别结果 ===");
        Console.WriteLine($"文本：{result.Text ?? string.Empty}");
        Console.WriteLine($"长度：{result.TextLen}");
        Console.WriteLine();

        if (result.Tokens == null || result.Tokens.Count == 0 || result.Timestamps == null || result.Timestamps.Count == 0)
        {
            Console.WriteLine("无可用的详细时间戳信息。");
            return;
        }

        Console.WriteLine("逐词时间戳明细：");
        Console.WriteLine($"{"#",3} {"Token",-20} {"开始 (s)",10} {"结束 (s)",10} {"时长 (ms)",12}");

        int count = Math.Min(result.Tokens.Count, result.Timestamps.Count);
        for (int i = 0; i < count; i++)
        {
            string token = result.Tokens[i];
            int[] stamp = result.Timestamps[i];
            if (stamp == null || stamp.Length == 0)
            {
                continue;
            }
            double startMs = stamp[0];
            double endMs = stamp[stamp.Length - 1];
            double durationMs = Math.Max(0, endMs - startMs);
            Console.WriteLine(
                $"{i,3} {token,-20} {startMs / 1000.0,10:F3} {endMs / 1000.0,10:F3} {durationMs,12:F1}");
        }

        Console.WriteLine();
    }

    private static void PrintPerformance(TimeSpan elapsed, TimeSpan audioDuration)
    {
        double audioSeconds = audioDuration.TotalSeconds;
        double rtf = audioSeconds <= 0 ? double.NaN : elapsed.TotalSeconds / audioSeconds;
        Console.WriteLine("=== 性能统计 ===");
        Console.WriteLine($"音频时长：{audioDuration.TotalSeconds:F2} s");
        Console.WriteLine($"识别耗时：{elapsed.TotalSeconds:F2} s");
        Console.WriteLine($"实时率 (RTF)：{rtf.ToString("F3", CultureInfo.InvariantCulture)}x");
    }

    private sealed record ModelFileSet(
        string ModelFile,
        string ConfigFile,
        string MvnFile,
        string TokensFile,
        string? ModelebFile,
        string? HotwordFile);

    private sealed class ModelFileResolver
    {
        public static ModelFileSet Resolve(RecognitionOptions options)
        {
            if (!Directory.Exists(options.ModelDirectory))
            {
                throw new DirectoryNotFoundException(options.ModelDirectory);
            }

            var files = Directory.GetFiles(options.ModelDirectory, "*", SearchOption.TopDirectoryOnly)
                .Select(path => new FileInfo(path))
                .ToList();

            string modelFile = RequireFile("模型 ONNX", files, file =>
                    file.Name.StartsWith("model", StringComparison.OrdinalIgnoreCase) &&
                    !file.Name.Contains("_eb", StringComparison.OrdinalIgnoreCase) &&
                    file.Extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase),
                options.Accuracy);

            string? modelebFile = null;
            if (options.ModelType.Equals("seacoparaformer", StringComparison.OrdinalIgnoreCase))
            {
                modelebFile = RequireFile("热词 Embed ONNX", files,
                    file => file.Name.StartsWith("model_eb", StringComparison.OrdinalIgnoreCase) &&
                            file.Extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase),
                    options.Accuracy);
            }

            string configFile = RequireFile("配置文件 (asr.yaml/json)", files, file =>
                file.Name.StartsWith("asr", StringComparison.OrdinalIgnoreCase) &&
                (file.Extension.Equals(".yaml", StringComparison.OrdinalIgnoreCase) ||
                 file.Extension.Equals(".yml", StringComparison.OrdinalIgnoreCase) ||
                 file.Extension.Equals(".json", StringComparison.OrdinalIgnoreCase)));

            string mvnFile = RequireFile("MVN 文件 (am.mvn)", files, file =>
                file.Name.StartsWith("am", StringComparison.OrdinalIgnoreCase) &&
                file.Extension.Equals(".mvn", StringComparison.OrdinalIgnoreCase));

            string tokensFile = RequireFile("词表文件 (tokens.txt)", files, file =>
                file.Name.StartsWith("tokens", StringComparison.OrdinalIgnoreCase) &&
                file.Extension.Equals(".txt", StringComparison.OrdinalIgnoreCase));

            string? hotwordFile = FindFile(files, file =>
                file.Name.StartsWith("hotword", StringComparison.OrdinalIgnoreCase) &&
                file.Extension.Equals(".txt", StringComparison.OrdinalIgnoreCase));

            return new ModelFileSet(
                ModelFile: modelFile,
                ConfigFile: configFile,
                MvnFile: mvnFile,
                TokensFile: tokensFile,
                ModelebFile: modelebFile,
                HotwordFile: options.ModelType.Equals("seacoparaformer", StringComparison.OrdinalIgnoreCase)
                    ? hotwordFile
                    : null);
        }

        private static string RequireFile(
            string description,
            List<FileInfo> files,
            Func<FileInfo, bool> predicate,
            string? accuracy = null)
        {
            string? path = FindFile(files, predicate, accuracy);
            if (path == null)
            {
                throw new FileNotFoundException($"未在模型目录中找到 {description}。");
            }
            return path;
        }

        private static string? FindFile(
            List<FileInfo> files,
            Func<FileInfo, bool> predicate,
            string? accuracy = null)
        {
            var candidates = files.Where(predicate).ToList();
            if (candidates.Count == 0)
            {
                return null;
            }

            if (!string.IsNullOrWhiteSpace(accuracy))
            {
                var preferred = candidates.LastOrDefault(f =>
                    f.Name.Contains($".{accuracy}.", StringComparison.OrdinalIgnoreCase));
                if (preferred != null)
                {
                    return preferred.FullName;
                }
            }
            return candidates.Last().FullName;
        }
    }

    private sealed class RecognitionOptions
    {
        private RecognitionOptions(
            string modelType,
            string modelDirectory,
            string audioPath,
            string accuracy,
            int threads)
        {
            ModelType = modelType;
            ModelDirectory = modelDirectory;
            AudioPath = audioPath;
            Accuracy = accuracy;
            Threads = threads;
        }

        public string ModelType { get; }
        public string ModelDirectory { get; }
        public string AudioPath { get; }
        public string Accuracy { get; }
        public int Threads { get; }

        public static RecognitionOptions Parse(string[] args)
        {
            if (args.Length == 0)
            {
                throw new ArgumentException("未提供任何参数。");
            }

            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];
                if (arg.StartsWith("--", StringComparison.Ordinal))
                {
                    string key = arg;
                    if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                    {
                        throw new ArgumentException($"参数 {key} 缺少取值。");
                    }
                    map[key] = args[++i];
                }
            }

            if (!map.TryGetValue("--model", out string? modelType))
            {
                throw new ArgumentException("必须指定 --model");
            }
            if (!map.TryGetValue("--model-dir", out string? modelDir))
            {
                throw new ArgumentException("必须指定 --model-dir");
            }
            if (!map.TryGetValue("--audio", out string? audio))
            {
                throw new ArgumentException("必须指定 --audio");
            }

            map.TryGetValue("--accuracy", out string? accuracy);
            map.TryGetValue("--threads", out string? threadsValue);

            int threads = 2;
            if (!string.IsNullOrWhiteSpace(threadsValue) &&
                !int.TryParse(threadsValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out threads))
            {
                throw new ArgumentException("--threads 需要整数");
            }

            accuracy ??= "int8";

            return new RecognitionOptions(
                modelType: modelType.Trim(),
                modelDirectory: modelDir.Trim(),
                audioPath: audio.Trim(),
                accuracy: accuracy.Trim(),
                threads: Math.Max(1, threads));
        }

        public static void PrintUsage()
        {
            Console.WriteLine();
            Console.WriteLine("用法：");
            Console.WriteLine("  dotnet run --project examples/ManySpeech.ParaformerOfflineTest -- " +
                              "--model paraformer|sensevoicesmall|seacoparaformer " +
                              "--model-dir <模型目录> --audio <音频文件> [--accuracy int8] [--threads 2]");
            Console.WriteLine();
            Console.WriteLine("示例：");
            Console.WriteLine("  dotnet run --project examples/ManySpeech.ParaformerOfflineTest -- " +
                              "--model seacoparaformer --model-dir D:/models/seaco --audio sample.wav " +
                              "--accuracy int8 --threads 4");
            Console.WriteLine();
        }
    }
}
