using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
using PreProcessUtils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace ManySpeech.ParaformerOfflineTest;

internal static class Program
{
    private static readonly ModelConfiguration[] ModelConfigurations =
    {
        new(
            Key: "paraformer",
            DisplayName: "Paraformer 通用离线",
            ModelType: "paraformer",
            ModelDirectory: "/ModelFiles/ASR/paraformer-large-zh-en-onnx-offline",
            Accuracy: "int8",
            Threads: 2,
            EnableItn: false,
            EnablePunctuation: false),
        new(
            Key: "sensevoicesmall",
            DisplayName: "SenseVoice Small",
            ModelType: "sensevoicesmall",
            ModelDirectory: "/ModelFiles/ASR/sensevoice-small-onnx",
            Accuracy: "int8",
            Threads: 2,
            EnableItn: true,
            EnablePunctuation: true),
        new(
            Key: "seacoparaformer",
            DisplayName: "Paraformer Seaco 热词",
            ModelType: "seacoparaformer",
            ModelDirectory: "/ModelFiles/ASR/paraformer-seaco-large-zh-timestamp-onnx-offline",
            Accuracy: "int8",
            Threads: 2,
            EnableItn: false,
            EnablePunctuation: false)
    };

    public static int Main(string[] args)
    {
        IReadOnlyList<ModelEntry> modelEntries = Array.Empty<ModelEntry>();
        try
        {
            modelEntries = InitializeModels();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("模型初始化失败：");
            Console.Error.WriteLine(ex);
            return 1;
        }

        try
        {
            RunInteractiveLoop(modelEntries);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("识别过程中发生异常：");
            Console.Error.WriteLine(ex);
            return 1;
        }
        finally
        {
            foreach (ModelEntry entry in modelEntries)
            {
                entry.Recognizer.Dispose();
            }
        }
    }

    private static IReadOnlyList<ModelEntry> InitializeModels()
    {
        List<ModelEntry> entries = new();
        Console.WriteLine("=== 初始化模型 ===");
        foreach (ModelConfiguration config in ModelConfigurations)
        {
            try
            {
                if (!Directory.Exists(config.ModelDirectory))
                {
                    throw new DirectoryNotFoundException(config.ModelDirectory);
                }

                var fileSet = ModelFileResolver.Resolve(
                    config.ModelDirectory,
                    config.ModelType,
                    config.Accuracy);

                var recognizer = CreateRecognizer(fileSet, config.Threads);

                bool useItn = config.EnableItn && fileSet.SupportsItn;
                bool usePunctuation = config.EnablePunctuation && fileSet.SupportsPunctuation;

                if (useItn)
                {
                    recognizer.ConfigureRuntimeOptions(useItn: true);
                }

                entries.Add(new ModelEntry(config, fileSet, recognizer, useItn, usePunctuation));
                Console.WriteLine($"[OK] {config.DisplayName} ({config.ModelType}) " +
                                  $"ITN:{(useItn ? "ON" : "OFF")} 标点:{(usePunctuation ? "ON" : "OFF")}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[跳过] {config.DisplayName}：{ex.Message}");
            }
        }

        if (entries.Count == 0)
        {
            throw new InvalidOperationException("未能成功加载任何模型，请检查配置路径。");
        }

        Console.WriteLine();
        return entries;
    }

    private static OfflineRecognizer CreateRecognizer(ModelFileSet fileSet, int threads)
    {
        return new OfflineRecognizer(
            modelFilePath: fileSet.ModelFile,
            configFilePath: fileSet.ConfigFile,
            mvnFilePath: fileSet.MvnFile,
            tokensFilePath: fileSet.TokensFile,
            modelebFilePath: fileSet.ModelebFile ?? string.Empty,
            hotwordFilePath: fileSet.HotwordFile ?? string.Empty,
            batchSize: 1,
            threadsNum: threads);
    }

    private static void RunInteractiveLoop(IReadOnlyList<ModelEntry> modelEntries)
    {
        while (true)
        {
            ModelEntry? entry = PromptModelSelection(modelEntries);
            if (entry == null)
            {
                Console.WriteLine("退出程序。");
                return;
            }

            RunRecognitionLoop(entry);
        }
    }

    private static ModelEntry? PromptModelSelection(IReadOnlyList<ModelEntry> modelEntries)
    {
        Console.WriteLine("=== 可用模型 ===");
        for (int i = 0; i < modelEntries.Count; i++)
        {
            ModelEntry entry = modelEntries[i];
            Console.WriteLine($"{i + 1}. {entry.Configuration.DisplayName} ({entry.Configuration.ModelType})");
            Console.WriteLine($"    目录：{entry.Configuration.ModelDirectory}");
        }
        Console.WriteLine("输入序号选择模型，直接回车退出：");

        string? input = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(input))
        {
            return null;
        }

        if (!int.TryParse(input.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int index) ||
            index < 1 || index > modelEntries.Count)
        {
            Console.WriteLine("无效的序号，请重试。");
            Console.WriteLine();
            return PromptModelSelection(modelEntries);
        }

        Console.WriteLine();
        return modelEntries[index - 1];
    }

    private static void RunRecognitionLoop(ModelEntry entry)
    {
        Console.WriteLine($"已选择模型：{entry.Configuration.DisplayName}");
        Console.WriteLine($"当前 ITN：{(entry.UseItn ? "开启" : entry.FileSet.SupportsItn ? "未开启" : "不支持")}");
        Console.WriteLine($"当前标点：{(entry.UsePunctuation ? "开启" : entry.FileSet.SupportsPunctuation ? "未开启" : "不支持")}");
        Console.WriteLine("输入音频文件路径开始识别，直接回车返回模型列表。");

        while (true)
        {
            Console.Write("音频文件> ");
            string? audioPath = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(audioPath))
            {
                Console.WriteLine();
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
                RunRecognition(entry, audioPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine("识别失败：");
                Console.WriteLine(ex.Message);
            }
        }
    }

    private static void RunRecognition(ModelEntry entry, string audioPath)
    {
        TimeSpan audioDuration = TimeSpan.Zero;
        float[][] channelSamples = AudioHelper.GetFileChannelSamples(audioPath, ref audioDuration);
        if (channelSamples.Length == 0 || channelSamples.All(samples => samples.Length == 0) || audioDuration.TotalMilliseconds <= 0)
        {
            throw new InvalidOperationException("音频文件为空或无法确定时长");
        }

        ReportConfiguration(entry, audioPath, channelSamples.Length);

        var combinedText = new StringBuilder();
        for (int channelIndex = 0; channelIndex < channelSamples.Length; channelIndex++)
        {
            float[] samples = channelSamples[channelIndex];
            if (samples.Length == 0)
            {
                Console.WriteLine($"--- 通道 {channelIndex + 1} ---");
                Console.WriteLine("该通道未检测到有效音频数据，已跳过。");
                Console.WriteLine();
                continue;
            }

            Console.WriteLine($"--- 通道 {channelIndex + 1} ---");
            var (result, elapsed) = RecognizeSamples(entry.Recognizer, samples);
            PrintResult(result);
            PrintPerformance(elapsed, audioDuration);
            if (!string.IsNullOrWhiteSpace(result.Text))
            {
                if (combinedText.Length > 0)
                {
                    combinedText.AppendLine();
                }
                combinedText.Append(result.Text);
            }
        }

        if (channelSamples.Length > 1)
        {
            Console.WriteLine("=== 通道合并文本 ===");
            Console.WriteLine(combinedText.ToString().TrimEnd());
            Console.WriteLine();
        }
    }

    private static (OfflineRecognizerResultEntity Result, TimeSpan Elapsed) RecognizeSamples(OfflineRecognizer recognizer, float[] samples)
    {
        using var stream = recognizer.CreateOfflineStream();
        var watch = Stopwatch.StartNew();
        stream.AddSamples(samples);
        OfflineRecognizerResultEntity result = recognizer.GetResult(stream);
        watch.Stop();
        return (result, watch.Elapsed);
    }

    private static void ReportConfiguration(ModelEntry entry, string audioPath, int channelCount)
    {
        ModelFileSet fileSet = entry.FileSet;
        ModelConfiguration config = entry.Configuration;

        Console.WriteLine("=== 识别配置 ===");
        Console.WriteLine($"模型名称：{config.DisplayName}");
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
        Console.WriteLine($"线程数：{config.Threads}");
        Console.WriteLine($"通道数：{channelCount}");

        string itnStatus = entry.UseItn
            ? "已启用"
            : fileSet.SupportsItn
                ? "未启用"
                : "不支持";
        string puncStatus = entry.UsePunctuation
            ? "已启用"
            : fileSet.SupportsPunctuation
                ? "未启用"
                : "不支持";
        Console.WriteLine($"逆文本正则化 (ITN)：{itnStatus}");
        Console.WriteLine($"标点恢复：{puncStatus}");
        Console.WriteLine();
    }

    private static void PrintResult(OfflineRecognizerResultEntity result)
    {
        Console.WriteLine("=== 识别结果 ===");
        string text = result.Text ?? string.Empty;
        string normalized = text.Replace("\r\n", "\n").Replace('\r', '\n');
        string[] segments = normalized.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        if (segments.Length <= 1)
        {
            Console.WriteLine($"文本：{text}");
        }
        else
        {
            Console.WriteLine($"段数：{segments.Length}");
            for (int i = 0; i < segments.Length; i++)
            {
                Console.WriteLine($"[{i + 1}] {segments[i].Trim()}");
            }
        }

        Console.WriteLine($"长度：{result.TextLen}");
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

    private sealed record ModelConfiguration(
        string Key,
        string DisplayName,
        string ModelType,
        string ModelDirectory,
        string Accuracy,
        int Threads,
        bool EnableItn = false,
        bool EnablePunctuation = false);

    private sealed record ModelEntry(
        ModelConfiguration Configuration,
        ModelFileSet FileSet,
        OfflineRecognizer Recognizer,
        bool UseItn,
        bool UsePunctuation);

    private sealed record ModelFileSet(
        string ModelType,
        string ModelDirectory,
        string Accuracy,
        string ModelFile,
        string ConfigFile,
        string MvnFile,
        string TokensFile,
        string? ModelebFile,
        string? HotwordFile,
        bool SupportsItn,
        bool SupportsPunctuation);

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

            bool supportsItn = false;
            bool supportsPunctuation = false;
            if (!string.IsNullOrEmpty(configFile) && File.Exists(configFile))
            {
                try
                {
                    string configContent = File.ReadAllText(configFile);
                    supportsItn = configContent.IndexOf("use_itn", StringComparison.OrdinalIgnoreCase) >= 0;
                }
                catch
                {
                    // ignore parsing issues
                }
            }

            if (modelType.Equals("sensevoicesmall", StringComparison.OrdinalIgnoreCase))
            {
                supportsItn = true;
                supportsPunctuation = true;
            }
            else
            {
                try
                {
                    supportsPunctuation = Directory.EnumerateFiles(modelDirectory, "*punc*", SearchOption.TopDirectoryOnly).Any();
                }
                catch
                {
                    supportsPunctuation = false;
                }
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
                HotwordFile: hotwordFile,
                SupportsItn: supportsItn,
                SupportsPunctuation: supportsPunctuation);
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
