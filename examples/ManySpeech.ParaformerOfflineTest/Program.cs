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
    private static class AppConfiguration
    {
        public const string ModelType = "paraformer";
        public const string ModelDirectory = "/ModelFiles/ASR/paraformer-large-zh-en-onnx-offline";
        public const string Accuracy = "int8";
        public const int Threads = 2;
    }

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
            ValidateModelConfiguration();

            var fileSet = ModelFileResolver.Resolve(
                AppConfiguration.ModelDirectory,
                AppConfiguration.ModelType,
                AppConfiguration.Accuracy);

            Console.WriteLine("=== 模型初始化 ===");
            Console.WriteLine($"模型目录：{fileSet.ModelDirectory}");
            Console.WriteLine($"模型类型：{fileSet.ModelType}");
            Console.WriteLine($"精度优先级：{fileSet.Accuracy}");

            using var recognizer = CreateRecognizer(fileSet);

            Console.WriteLine("模型加载完毕，输入音频文件路径开始识别，直接回车退出。");
            RunInteractiveLoop(recognizer, fileSet);
            return 0;
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"参数错误：{ex.Message}");
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

    private static void ValidateModelConfiguration()
    {
        if (!SupportedModels.Contains(AppConfiguration.ModelType, StringComparer.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"暂不支持的模型类型：{AppConfiguration.ModelType}，可选值：{string.Join(", ", SupportedModels)}");
        }
        if (!Directory.Exists(AppConfiguration.ModelDirectory))
        {
            throw new DirectoryNotFoundException(AppConfiguration.ModelDirectory);
        }
    }

    private static OfflineRecognizer CreateRecognizer(ModelFileSet fileSet)
    {
        return new OfflineRecognizer(
            modelFilePath: fileSet.ModelFile,
            configFilePath: fileSet.ConfigFile,
            mvnFilePath: fileSet.MvnFile,
            tokensFilePath: fileSet.TokensFile,
            modelebFilePath: fileSet.ModelebFile ?? string.Empty,
            hotwordFilePath: fileSet.HotwordFile ?? string.Empty,
            batchSize: 1,
            threadsNum: AppConfiguration.Threads);
    }

    private static void RunInteractiveLoop(OfflineRecognizer recognizer, ModelFileSet fileSet)
    {
        while (true)
        {
            Console.Write("音频文件> ");
            string? audioPath = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(audioPath))
            {
                Console.WriteLine("退出程序。");
                return;
            }

            audioPath = audioPath.Trim();
            if (!File.Exists(audioPath))
            {
                Console.WriteLine("文件不存在，请重新输入。");
                continue;
            }

            try
            {
                RunRecognition(recognizer, fileSet, audioPath);
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine("识别失败：");
                Console.WriteLine(ex);
            }
        }
    }

    private static void RunRecognition(OfflineRecognizer recognizer, ModelFileSet fileSet, string audioPath)
    {
        ReportConfiguration(fileSet, audioPath, AppConfiguration.Threads);

        TimeSpan audioDuration = TimeSpan.Zero;
        float[] samples = AudioHelper.GetFileSample(audioPath, ref audioDuration);
        if (samples.Length == 0 || audioDuration.TotalMilliseconds <= 0)
        {
            throw new InvalidOperationException("音频文件为空或无法确定时长");
        }

        using var stream = recognizer.CreateOfflineStream();

        var watch = Stopwatch.StartNew();
        stream.AddSamples(samples);
        OfflineRecognizerResultEntity result = recognizer.GetResult(stream);
        watch.Stop();

        PrintResult(result);
        PrintPerformance(watch.Elapsed, audioDuration);
    }

    private static void ReportConfiguration(ModelFileSet fileSet, string audioPath, int threads)
    {
        Console.WriteLine("=== 识别配置 ===");
        Console.WriteLine($"模型类型：{fileSet.ModelType}");
        Console.WriteLine($"模型目录：{fileSet.ModelDirectory}");
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
        Console.WriteLine($"模型精度偏好：{fileSet.Accuracy}");
        Console.WriteLine($"音频文件：{audioPath}");
        Console.WriteLine($"线程数：{threads}");
        Console.WriteLine();
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
        string ModelType,
        string ModelDirectory,
        string Accuracy,
        string ModelFile,
        string ConfigFile,
        string MvnFile,
        string TokensFile,
        string? ModelebFile,
        string? HotwordFile);

    private sealed class ModelFileResolver
    {
        public static ModelFileSet Resolve(string modelDirectory, string modelType, string? accuracy)
        {
            if (!Directory.Exists(modelDirectory))
            {
                throw new DirectoryNotFoundException(modelDirectory);
            }

            string? accuracyNormalized = string.IsNullOrWhiteSpace(accuracy) ? null : accuracy.Trim();
            bool isSeaco = modelType.Equals("seacoparaformer", StringComparison.OrdinalIgnoreCase);

            var files = Directory.GetFiles(modelDirectory, "*", SearchOption.TopDirectoryOnly)
                .Select(path => new FileInfo(path))
                .ToList();

            string modelFile = RequireFile("模型 ONNX", files, file =>
                    file.Name.StartsWith("model", StringComparison.OrdinalIgnoreCase) &&
                    !file.Name.Contains("_eb", StringComparison.OrdinalIgnoreCase) &&
                    file.Extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase),
                accuracyNormalized);

            string? modelebFile = null;
            if (isSeaco)
            {
                modelebFile = RequireFile("热词 Embed ONNX", files,
                    file => file.Name.StartsWith("model_eb", StringComparison.OrdinalIgnoreCase) &&
                            file.Extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase),
                    accuracyNormalized);
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

            string? hotwordFile = null;
            if (isSeaco)
            {
                hotwordFile = FindFile(files, file =>
                    file.Name.StartsWith("hotword", StringComparison.OrdinalIgnoreCase) &&
                    file.Extension.Equals(".txt", StringComparison.OrdinalIgnoreCase));
            }

            return new ModelFileSet(
                ModelType: modelType,
                ModelDirectory: modelDirectory,
                Accuracy: accuracyNormalized ?? string.Empty,
                ModelFile: modelFile,
                ConfigFile: configFile,
                MvnFile: mvnFile,
                TokensFile: tokensFile,
                ModelebFile: modelebFile,
                HotwordFile: hotwordFile);
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
}
