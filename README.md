# GVS2 — 基于视觉 LLM 的字幕样式匹配工具

> **G**enerative **V**ision **S**ubtitle — 第二代

GVS2 是一个利用视觉大语言模型（Vision LLM）从视频中识别硬字幕样式并自动回写 ASS 字幕文件的桌面工具。

本项目受到 [jianchang512/gvs](https://github.com/jianchang512/gvs) 的启发，在其"用视觉 LLM 替代传统 OCR 读取硬字幕"的核心思路上进行了大幅改进和重构。

---

## 项目定位

在字幕处理工作流中，GVS2 旨在衔接以下两个工具之间的空白环节：

```
[MagiaTimeline]          -->   [GVS2]         -->   [VideoCaptioner]
提取硬字幕时间轴(.ass)         识别字幕样式并回写            LLM 字幕翻译
```

- **[MagiaTimeline](https://github.com/HurryPeng/MagiaTimeline)**：基于计算机视觉从视频中提取硬字幕的精确时间轴，生成 `.ass` 文件。擅长检测字幕出现/消失的时间点。虽然也能处理字幕文本内容和样式，但样式分类质量不够稳定，文本识别精度与 [video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor) 相当。
- **[VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)**：LLM 驱动的字幕助手，负责语音识别、字幕断句优化、翻译和视频合成。专注于文本翻译与质量优化，不涉及视觉样式识别。
- **GVS2**：填补中间环节——当字幕时间轴已有但样式信息缺失时，用视觉 LLM 逐帧识别字幕的视觉样式（颜色、描边、行数等），并将匹配结果写回 ASS 的 `Style` 字段。

---

## 与 [video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor) 的对比

| 维度 | video-subtitle-extractor (VSE) | GVS2 |
|------|-------------------------------|------|
| **核心技术** | PaddleOCR 本地 OCR | 云端视觉 LLM（豆包、千问、MiMo、GLM、GPT、Claude 等） |
| **是否需要 GPU** | 推荐（支持 CUDA/DirectML/ONNX） | 不需要 |
| **离线可用** | 完全离线 | 需要网络和 API Key |
| **字幕样式识别** | 不支持，仅提取纯文本 | 核心功能，可识别颜色、描边、行数等 |
| **时间轴来源** | 自己从视频逐帧检测 | 需要外部提供（如 MagiaTimeline 生成的 ASS） |
| **输出格式** | SRT / 纯文本 | ASS（带样式回写） |
| **多语言** | 87 种语言（OCR 模型） | 依赖 LLM 能力，主流语言均可 |
| **处理速度** | 快速模式较快，精确模式很慢 | 并发请求，344 条字幕约 7-8 分钟 |
| **路径限制** | 不能含中文字符或空格 | 无此限制 |
| **准确率** | 快速模式可能漏行、有错字 | 依赖 LLM，文本提取准确率高 |

**GVS2 的优势：**
- 能识别字幕的视觉样式（VSE 完全不具备此能力）
- 不需要 GPU，部署门槛低
- 路径无特殊限制
- 支持并发处理，可配置并发数
- 支持多样式分类（同一视频中不同说话人用不同字幕样式）

**GVS2 的劣势：**
- 需要网络和 API Key，有调用成本
- 不具备独立的时间轴提取能力，需要配合 MagiaTimeline 等工具
- 不具备语音识别能力，不能从音频生成字幕
- 处理速度受限于 API 响应时间

---

## 工作流程

```
视频文件 (.mp4)
    |
    v
[字幕时间轴] ← 已有 ASS/SRT（可由 MagiaTimeline 生成）
    |
    v
[逐事件截图] ← 按字幕事件的时间点，用 FFmpeg 截取帧
    |
    v
[图像预处理] ← 裁切字幕区域、缩放、编码
    |
    v
[视觉 LLM]  ← 并发发送到 OpenAI-compatible / Anthropic API
    |
    ├── 样式识别：判断属于哪一种已锁定样式
    └── 文字提取：读取硬字幕文本
    |
    v
[ASS 写回]   ← 将样式和文字结果写入 ASS 文件
    |
    v
输出 ASS 文件
```

---

## 快速开始

### 环境要求

- Python 3.10+
- FFmpeg（需要在 PATH 中）
- 网络连接（调用云端 LLM API）

### 安装

```bash
pip install -r requirements.txt
```

或使用 uv：

```bash
uv sync
```

### 启动主界面

```bash
python app.py
```

### 启动 Benchmark 界面

```bash
python benchmark_app.py
```

命令行 benchmark 模式：

```bash
python benchmark_app.py benchmark_samples.example.json
```

---

## 配置说明

### 方式一：界面配置

启动主界面后，通过顶部菜单 `设置 -> API 设置` 填写 provider 配置，保存后写入 `config.json`。

### 方式二：环境变量

设置环境变量作为默认值，界面为空时自动补全：

```bash
# OpenAI-compatible provider
export GVS2_OPENAI_BASE_URL="https://api.openai.com/v1"
export GVS2_OPENAI_API_KEY="your-key"
export GVS2_OPENAI_MODEL="gpt-4o"

# Anthropic provider
export GVS2_ANTHROPIC_BASE_URL="https://api.anthropic.com"
export GVS2_ANTHROPIC_API_KEY="your-key"
export GVS2_ANTHROPIC_MODEL="claude-sonnet-4-6"
```

Windows 持久化：

```powershell
setx GVS2_OPENAI_BASE_URL "https://api.openai.com/v1"
setx GVS2_OPENAI_API_KEY "your-key"
setx GVS2_OPENAI_MODEL "gpt-4o"
```

### config.json 示例

参见 `config.example.json`。复制为 `config.json` 后填入你的配置：

```bash
cp config.example.json config.json
```

---

## 样式配置

GVS2 的核心能力是识别视频中硬字幕的视觉样式。使用前需要先锁定目标字幕样式：

1. 在主界面加载视频和字幕文件
2. 播放视频，定位到有字幕的画面，设置好字幕区域范围
3. 配置样式识别任务的 API，调整预览图尺寸和压缩质量，确保肉眼可以辨认字幕内容
4. 参照 `styles_example.json` 填写字幕特征描述。如不确定如何填写，可点击”自然语言转紧凑描述”输入中文进行转换，比”从预览图生成”更节省 Token
5. 也可以直接点击”从预览图生成样式描述”，程序将调用 LLM 分析字幕外观并输出结构化描述
6. 编辑当前硬字幕样式的紧凑描述文本（颜色、描边、行数等特征）
7. 重复以上步骤，直到覆盖所有需要识别的字幕样式
8. 可导出为 JSON 模板，下次直接导入

样式配置示例参见 `styles_example.json`。

---

## 支持的 Provider

| Provider 类型 | 代表模型 | 说明 |
|--------------|---------|------|
| OpenAI-compatible | GPT、豆包、智谱、小米 MiMo、Kimi 等 | 通用接口，兼容大部分国内外模型 |
| Anthropic | Claude Haiku 4.5 | 原生 Anthropic API |

两个任务（样式识别、文字提取）可以分别配置不同的 provider。

**成本提示**：由于本工具处理的视频数据通常已经公开或即将公开至互联网，可以考虑通过 Codex、Open Code Go、火山方舟协作计划等渠道调用 API 以降低成本。例如，火山方舟协作计划支持同时开通多个版本的 Doubao-seed 模型，灵活切换可进一步优化费用。

---

## 性能与成本估算

以下数据基于实测日志（已去除个人信息）：

### 纯文字提取

| 指标 | 数值 |
|------|------|
| 字幕条数 | 344 条 |
| 并发数 | 4 |
| 总耗时 | ~7 分 20 秒 |
| 成功提取 | 336 条 (97.7%) |
| 跳过（无字幕区域） | 8 条 (2.3%) |

> 以 `doubao-seed-2-0-mini-260215` 为例，344 条字幕对应的截图约消耗 500k Token。

### 样式分类

（待补充）

---

## 主界面功能

### 输入输出
- 视频输入（mp4 等常见格式）
- ASS / SRT 字幕输入
- ASS 字幕输出

### 样式整理
- 自然语言描述转紧凑样式标签
- 预览图自动样式分析
- 样式识别 prompt 预览弹窗
- 中文样式关键词前置拦截

### 样式锁定
- 样式表格编辑与自动编号
- JSON 模板导入导出
- 从当前预览画面锁定样式

### 预览与交互
- 视频预览与字幕区域遮罩
- 百分比字幕区域配置
- 字幕语言选择（自动 / 简中 / 繁中 / 日语 / 英语 / 韩语 / 混合）
- 上一个 / 下一个字幕时点导航
- 单次文字识别测试

### 运行能力
- 样式识别任务（可独立启用/禁用）
- 文字提取任务（可独立启用/禁用）
- 两个任务可配置不同 provider
- 可配置并发数、超时时间、图像压缩参数

---

## Benchmark 功能

- CLI 和 GUI 两种模式
- JSON 样本导入与预览
- 汇总表格（样式准确率、文字精确匹配率、文字相似度、Token 消耗、费用估算）
- 支持按自定义名称或模型名分组统计
- 表格文本导出

---

## 项目创新点

1. **视觉 LLM 替代传统 OCR**：不依赖本地 OCR 模型和 GPU，通过云端视觉大模型实现更灵活的字幕识别，尤其对复杂字体、艺术字、描边字幕有更好的识别能力。

2. **样式识别与文字提取分离**：将"字幕长什么样"和"字幕写了什么"拆分为两个独立任务，可分别配置不同模型和参数，灵活度高。

3. **事件驱动而非帧驱动**：不按固定帧率逐帧扫描，而是按字幕事件逐条处理，避免在无字幕帧上浪费 API 调用，显著降低成本。

4. **多样式锁定与分类**：支持同一视频中存在多种字幕样式（如不同说话人用不同颜色），通过 few-shot 样本图辅助 LLM 区分相似样式。

5. **需核查标记**：当 LLM 判断字幕画面存在歧义（如重叠文字、混合样式）时，自动标记为"需核查"，降低误判风险。

6. **模块化架构**：相比原版 GVS 的单文件设计，GVS2 将 Provider、Pipeline、Services、UI 分层解耦，便于扩展新的 LLM Provider 或处理逻辑。

7. **中文样式关键词前置拦截**：在将样式描述发送给 LLM 之前，先用本地规则清洗中文关键词，减少无效 API 调用。

---

## Benchmark 系统测试说明

> **当前状态：尚未完成系统化测试**

以下两类 Benchmark 尚未进行系统化的对比实验：

### 1. 图像清晰度 Benchmark

需要测试不同图像压缩参数对识别准确率的影响：
- `max_edge`（图像最大边长）：256 / 500 / 768 / 1024
- `quality`（压缩质量）：30 / 50 / 70 / 90
- `image_format`：WEBP vs JPEG

预期问题：图像过度压缩可能导致小字或模糊字幕识别失败。

### 2. 视觉模型 Benchmark

需要对比不同视觉 LLM 的识别效果和调用成本：

对比维度：
- 样式识别准确率
- 文字提取准确率
- Token 消耗量
- 单次调用成本

**欢迎使用者提供 Benchmark 数据和测试反馈。**

---

## 未来开发目标

### 短期
- [ ] 支持导入用户自定义的 ASS 样式模板，与锁定样式表自动对接
- [ ] 开发导入”标准答案”字幕与”识别结果”字幕后，自动提取 Benchmark 结构化数据的功能
- [ ] 完成图像清晰度 Benchmark 和视觉模型 Benchmark 系统测试
- [ ] 添加 Provider 表单及导入导出功能，记住历史使用过的 Provider 和模型以便快速切换
- [ ] 更严格地清洗样式整理模型的返回结果，避免多余文本污染 `feature_notes`
- [ ] 优化 SRT 转 ASS 的默认输出质量（PlayRes、字体尺寸、边距）

### 中期
- [ ] 样式样本库分类整理功能
- [ ] 支持更多 LLM Provider（如 Gemini、本地模型）
- [ ] 引入 [Aegisub](https://github.com/Aegisub/Aegisub) 的 ASS 预览功能，支持实时编辑字幕样式和文本内容
- [ ] 处理中断后恢复（断点续传）

### 长期
- [ ] 支持软字幕样式的自动映射
- [ ] 探索本地视觉模型的可行性，降低对云端 API 的依赖
- [ ] 探索利用 Whisper 等语音识别模型生成当前字幕事件的 ASR 字幕，作为 LLM 识别失败时的兜底方案
- [ ] 探索基于 Prompt 的 LLM 字幕纠正与翻译功能（也许是 GVS3 的课题了……）

---

## 需要使用者反馈的地方

1. **字幕区域配置**：当前使用百分比定义字幕区域，是否需要更直观的可视化配置方式？
2. **Benchmark 需求**：除了图像清晰度和模型对比，还需要哪些维度的测试？
3. **性能瓶颈**：在实际使用中，API 延迟和并发数的最佳平衡点在哪里？

---

## 当前限制

- SRT 输入会转换为带默认样式的 ASS 输出，更细的样式策略可继续打磨
- 未安装 PySide6 时无法启动 GUI
- 需要网络连接和 API Key

---

## 项目结构

```
GVS2/
├── app.py                  # 主入口
├── benchmark_app.py        # Benchmark 入口
├── config.example.json     # 配置模板
├── benchmark_samples.example.json  # Benchmark 样本示例
├── styles_example.json     # 样式配置示例
├── requirements.txt        # Python 依赖
├── pyproject.toml          # 项目元数据
├── pipeline/               # 处理管线
│   ├── runner.py           # GVS2 运行封装
│   └── event_pipeline.py   # 事件驱动的处理管线
├── providers/              # LLM Provider
│   ├── base.py             # 基础类型定义
│   ├── factory.py          # Provider 工厂
│   ├── openai_provider.py  # OpenAI-compatible 实现
│   ├── anthropic_provider.py # Anthropic 实现
│   ├── prompt_builder.py   # Prompt 构建
│   └── response_parser.py  # 响应解析
├── services/               # 核心服务
│   ├── subtitle_parser.py  # 字幕解析（ASS/SRT）
│   ├── ass_writer.py       # ASS 写入
│   ├── frame_sampler.py    # 帧采样
│   ├── image_preprocess.py # 图像预处理
│   ├── media.py            # 媒体工具（FFmpeg）
│   └── benchmark.py        # Benchmark 评分
├── models/                 # 数据模型
│   ├── subtitle_event.py   # 字幕事件
│   ├── style_profile.py    # 样式配置
│   └── job_result.py       # 任务结果
├── storage/                # 存储
│   └── settings_store.py   # 配置持久化
└── ui/                     # 界面
    ├── main_window.py      # 主窗口
    ├── benchmark_window.py # Benchmark 窗口
    └── widgets.py          # 通用组件
```

---

## 致谢

- [jianchang512/gvs](https://github.com/jianchang512/gvs) — 本项目的灵感来源，首次提出用视觉 LLM 读取硬字幕的思路
- [HurryPeng/MagiaTimeline](https://github.com/HurryPeng/MagiaTimeline) — 字幕时间轴提取框架
- [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) — LLM 字幕助手
- [YaoFANGUK/video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor) — 本地 OCR 字幕提取工具

---

## 许可证

MIT License
