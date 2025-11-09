# Paraformer 离线测试工具

一个面向 `paraformer` / `sensevoicesmall` / `seacoparaformer` 三类模型的简单离线识别脚本，用于验证模型文件是否可用，并给出实时率（RTF）与逐词时间戳。

## 使用方式

```bash
dotnet run --project examples/ManySpeech.ParaformerOfflineTest -- \
  --model paraformer \
  --model-dir D:/models/paraformer-large-zh \
  --audio sample.wav \
  --accuracy int8 \
  --threads 4
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--model` | 必填，模型类型，可选 `paraformer` / `sensevoicesmall` / `seacoparaformer` |
| `--model-dir` | 必填，模型文件所在目录，需包含 `model*.onnx`、`asr.yaml/json`、`am.mvn`、`tokens.txt` 等文件 |
| `--audio` | 必填，待识别的音频文件路径（支持 WAV/常见媒体，内部自动重采样至 16kHz 单声道） |
| `--accuracy` | 选填，优先选取包含该片段的模型文件（默认为 `int8`） |
| `--threads` | 选填，ONNX Runtime 推理线程数（默认 2） |

> `seacoparaformer` 需要额外的 `model_eb*.onnx` 和可选 `hotword*.txt`，脚本会自动匹配。

## 输出说明

- **识别配置**：展示实际使用的模型/配置/MVN/词表等文件位置
- **识别结果**：输出最终文本、长度，以及逐词时间戳（起止秒数与持续时长）
- **性能统计**：显示音频时长、识别耗时与实时率 `RTF = 推理耗时 / 音频时长`（<1 表示快于实时）

## 常见问题

- 若提示缺少文件，请检查模型目录是否包含必要文件，或调整 `--accuracy` 与模型类型。
- 若音频无法解析，请确认文件路径及格式；脚本内部使用 `PreProcessUtils.AudioHelper` 自动转码。


