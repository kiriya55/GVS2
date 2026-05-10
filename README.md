# GVS2

GVS2 是一个利用视觉大语言模型（Vision LLM）从视频中识别硬字幕样式及其内容，自动回写 ASS 字幕文件的桌面应用。

本项目在[GVS](https://github.com/jianchang512/gvs)"用视觉 LLM 替代传统 OCR 读取硬字幕"的核心思路上进行了大幅改进和重构。与 [video-subtitle-extractor (VSE)](https://github.com/YaoFANGUK/video-subtitle-extractor) 等工具相比，本项目不依赖本地 GPU 进行识别，也避免了部分字体下完全无法识别文本的情况。

GVS2 主要用于衔接 [MagiaTimeline](https://github.com/HurryPeng/MagiaTimeline)（打轴）到 [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)（翻译）之间的空白环节：匹配其他语种硬字幕样式，便于更快确定中文字幕对应的不遮挡原始字幕的样式，然后通过 ASS 源文件编辑或 Aegisub 批量修改等方式修改为对应样式。

此外，GVS2 也适用于识别字幕文本，主要适用场景为其他语种硬字幕固定出现于视频底部时的情况。

## 快速开始

运行 `install.bat`（适用于 Windows）或 `install.sh`（适用于 GNU/Linux 或 macOS），自动将 `requirements.txt` 中列出的依赖项安装到 Python 虚拟环境中。

然后通过运行 `app.bat` 或 `app.sh`（适用于 GNU/Linux 或 macOS）启动主界面，或运行 `benchmark.bat` / `benchmark.sh` 启动 benchmark 界面。

如果带一个 JSON 参数，则走命令行表格模式：

```bash
# Windows
benchmark.bat benchmark_samples.example.json

# GNU/Linux 或 macOS
./benchmark.sh benchmark_samples.example.json
```

## 依赖

- Python 3.10+
- FFmpeg（需加入 PATH）
- PySide6
- Pillow
- requests
- anthropic

## 配置方式

GVS2 支持两种配置来源：

- 在界面中直接填写 provider 配置，并保存到 `config.json`
- 用环境变量提供默认值，界面为空时自动补全

主界面中的 provider 配置入口为：顶部菜单 `设置 -> API 设置`，在弹出窗口中配置文字提取和样式识别两个任务的 LLM 参数、并发数、超时和图片处理选项。

当前支持两类 provider：
- OpenAI-compatible
- Anthropic

### 环境变量

GVS2 支持从环境变量自动读取 provider 默认值：

| 变量名 | 说明 |
|--------|------|
| `GVS2_OPENAI_BASE_URL` | OpenAI-compatible 接口地址 |
| `GVS2_OPENAI_API_KEY` | OpenAI-compatible API Key |
| `GVS2_OPENAI_MODEL` | OpenAI-compatible 模型名 |
| `GVS2_ANTHROPIC_BASE_URL` | Anthropic 接口地址 |
| `GVS2_ANTHROPIC_API_KEY` | Anthropic API Key |
| `GVS2_ANTHROPIC_MODEL` | Anthropic 模型名 |

Windows 持久化示例：

```powershell
setx GVS2_OPENAI_BASE_URL "https://api.openai.com/v1"
setx GVS2_OPENAI_API_KEY "your_openai_key"
setx GVS2_OPENAI_MODEL "gpt-4o"

setx GVS2_ANTHROPIC_BASE_URL "https://api.anthropic.com"
setx GVS2_ANTHROPIC_API_KEY "your_anthropic_key"
setx GVS2_ANTHROPIC_MODEL "claude-opus-4-7"
```

设置后重新打开终端即可生效。

## 主界面功能

### 输入输出

- 视频输入（mp4 / mkv / avi / mov）
- 字幕输入（ASS / SRT）
- ASS 输出

### 样式整理

- 自然语言转紧凑样式描述
- 预览图样式分析（从当前视频帧生成样式描述）
- 点击预览 prompt 弹窗查看样式识别 prompt
- 中文样式关键词前置拦截，避免把无关描述直接发给模型

### 样式锁定

- 样式表格编辑（样式ID / 显示名称 / ASS样式名 / 特征描述 / 布局提示）
- 自动编号
- 模板 JSON 导入导出
- 从 ASS 文件导入样式（解析 `[V4+ Styles]` 段）
- 追加当前紧凑描述为新样式
- 从当前预览锁定样式（自动保存样本图和元数据备份）

### 预览与交互

- 视频预览，可在画面上拖拽绘制红框字幕区域，并通过拖动框体、边缘或角点随时调整
- 字幕区域百分比信息（X / Y / W / H）
- 字幕语言选择（自动识别 / 简中 / 繁中 / 日语 / 英语 / 韩语 / 混合）
- **字幕事件跳转**：加载 ASS/SRT 后，可通过下拉框选择任意字幕事件直接跳转到对应时间点
- 上一个 / 下一个字幕时点快速导航
- 文字识别 dry run：直接测试当前预览区域的文字提取效果
- 预览处理图：查看 style_job 和 text_job 两个任务的预处理图片效果

### 使用方法

#### 样式分类

GVS2 的核心能力之一是识别视频中硬字幕的视觉样式。使用前需要先确定目标字幕样式：

1. 加载视频和字幕文件，通过预览找到目标字幕帧
2. 使用"从预览图生成样式描述"或手动输入自然语言描述，生成紧凑样式描述
3. 将样式锁定到样式表格中（可从 ASS 文件导入、手动添加、或从预览锁定）
4. 样式配置示例参见 `styles_example.json`（如 `aumi_styles.json`）

#### 文本识别

GVS2 的另一核心能力是提取视频中硬字幕的文本内容。支持自定义并发数、超时时间、图片格式和分辨率等参数。

当两个任务都启用时，将先执行样式分类再执行文字识别。

### 运行能力

- 样式识别任务
- 文字提取任务
- 两个任务可以分别配置不同 provider
- 两个任务至少启用一个
- 支持 few-shot 模式（使用已保存的样本图作为视觉参考）
- 失败事件清单导出与复跑

### 成本提示

由于本工具处理的视频数据通常已经公开或即将公开至互联网，可以考虑通过各类 API 协作计划或免费额度渠道调用 API 以降低成本。

以某个视觉模型为例，344 条字幕对应的截图约消耗 500k Token。具体费用取决于所用模型和 provider 定价。

### Provider 历史记录

- 保存常用 provider 配置到历史记录
- 快速切换不同 provider 配置

## benchmark 功能

- benchmark CLI / GUI
- JSON 样本预览
- 汇总表格与文本输出
- 支持按自定义名称或模型名分组统计

## 当前限制

- SRT 输入会转换为带基础 PlayRes、默认样式和边距策略的 ASS 输出
- 如果本地环境未安装 PySide6，则无法启动 GUI

## 开发目标

[MagiaTimeline](https://github.com/HurryPeng/MagiaTimeline) 识别的字幕仍然需要转换为srt后，通过其他软件确定是否存在错误时间轴，编辑后再导入 GVS2 进行识别。接下来的主要目标包括：

- 整合时间轴修改功能或新建时间轴修改脚本
- 更紧密的打轴与样式识别流程整合
- 更智能的样式匹配与自动分类
- 更准确的样式匹配效果
- 更多字幕场景的适配
