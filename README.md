# 🔬 Open Deep Research

Deep Research 已成为最受欢迎的智能体应用之一。本项目是一个简单、可配置、完全开源的深度研究智能体，支持多种模型提供商、搜索工具和 MCP 服务器。其性能与许多主流深度研究智能体相当（[参见 Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)）。

### 🔥 最近更新

- **2025年8月14日**：免费课程 [链接](https://academy.langchain.com/courses/deep-research-with-langgraph)（课程仓库 [链接](https://github.com/langchain-ai/deep_research_from_scratch)），介绍如何构建开源深度研究智能体。
- **2025年8月7日**：新增 GPT-5，并更新 Deep Research Bench 评测结果。
- **2025年8月2日**：在 [Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 获得第6名，总分 0.4344。
- **2025年7月30日**：项目演进详见[博客](https://rlancemartin.github.io/2025/07/30/bitter_lesson/)。
- **2025年7月16日**：更多信息见[博客](https://blog.langchain.com/open-deep-research/)和[视频](https://www.youtube.com/watch?v=agGiWUpxkhg)。

### 🚀 快速开始

1. 克隆仓库并激活虚拟环境：
	```bash
	git clone https://github.com/langchain-ai/open_deep_research.git
	cd open_deep_research
	uv venv
	source .venv/bin/activate  # Windows: .venv\Scripts\activate
	```

2. 安装依赖：
	```bash
	uv sync
	# 或
	uv pip install -r pyproject.toml
	```

3. 设置 `.env` 文件以自定义环境变量（模型选择、搜索工具等）：
	```bash
	cp .env.example .env
	```

4. 本地启动 LangGraph 服务器：
	```bash
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
	```
	这会在浏览器中打开 LangGraph Studio UI。

	```
	- 🚀 API: http://127.0.0.1:2024
	- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
	- 📚 API Docs: http://127.0.0.1:2024/docs
	```

	在 `messages` 输入框提问并点击 `Submit`，可在 "Manage Assistants" 标签页选择不同配置。

### ⚙️ 配置


#### LLM :brain:

支持多种 LLM 提供商，详情见 [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)。主要模型用途如下（详见 [configuration.py](src/open_deep_research/configuration.py)）：

- **摘要**（默认：`openai:gpt-4.1-mini`）：摘要搜索结果
- **研究**（默认：`openai:gpt-4.1`）：驱动搜索智能体
- **压缩**（默认：`openai:gpt-4.1`）：压缩研究发现
- **最终报告**（默认：`openai:gpt-4.1`）：撰写最终报告

> 注意：所选模型需支持结构化输出和工具调用。

> OpenRouter 和本地模型（Ollama）配置见相关指南。

#### 搜索 API :mag:

默认使用 [Tavily](https://www.tavily.com/) 搜索 API，支持 MCP 兼容和主流 LLM 原生网页搜索。配置详见 [configuration.py](src/open_deep_research/configuration.py)。

#### 其他

更多自定义设置详见 [configuration.py](src/open_deep_research/configuration.py)。

### 📊 评测

本项目已适配 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 基准，包含100个博士级研究任务（50英文，50中文），覆盖22个领域。排行榜基于 RACE 分数，采用 LLM 评审（Gemini）对比专家报告。

#### 使用方法

> 注意：完整评测约需 $20-$100，视模型而定。

数据集见 [LangSmith](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d)。运行评测：
```bash
python tests/run_evaluate.py
```
完成后，提取结果为 JSONL 文件：
```bash
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"
```
生成的 JSONL 文件可提交至 Deep Research Bench 仓库，详见其 [快速开始指南](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start)。

#### 结果示例

| 名称 | Commit | 摘要 | 研究 | 压缩 | 总花费 | 总 Token | RACE 分数 | 实验链接 |
|------|--------|------|------|------|--------|----------|-----------|----------|
| GPT-5 | [ca3951d](...) | openai:gpt-4.1-mini | openai:gpt-5 | openai:gpt-4.1 |  | 204,640,896 | 0.4943 | [Link](...) |
| 默认 | [6532a41](...) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98 | 58,015,332 | 0.4309 | [Link](...) |
| Claude Sonnet 4 | [f877ea9](...) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09 | 138,917,050 | 0.4401 | [Link](...) |
| Deep Research Bench 提交 | [c0a160b](...) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83 | 207,005,549 | 0.4344 | [Link](...) |

### 🚀 部署与使用

#### LangGraph Studio

参考 [快速开始](#-快速开始) 本地启动 LangGraph 服务器并在 Studio 测试。

#### 云端部署

可部署至 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。

#### Open Agent Platform

Open Agent Platform (OAP) 是面向非技术用户的智能体配置 UI。可在 OAP 公共演示实例中测试 Deep Researcher，只需添加 API Key。体验地址：[OAP](https://oap.langchain.com)

也可自行部署 OAP 并添加自定义智能体：
1. [部署 OAP](https://docs.oap.langchain.com/quickstart)
2. [添加 Deep Researcher](https://docs.oap.langchain.com/setup/agents)

### 旧版实现 🏛️

`src/legacy/` 文件夹包含两种早期实现，提供不同自动化研究思路，性能略逊于当前版本，但有助于理解深度研究的多种方法。

#### 1. 工作流实现 (`legacy/graph.py`)
- **计划-执行**：结构化工作流，支持人工规划
- **顺序处理**：逐步创建报告章节，带反思
- **交互控制**：可反馈和批准报告计划
- **质量导向**：强调准确性，迭代优化

#### 2. 多智能体实现 (`legacy/multi_agent.py`)
- **主管-研究员架构**：多智能体协作
- **并行处理**：多研究员同时工作
- **速度优化**：并发加速报告生成
- **MCP 支持**：广泛集成 Model Context Protocol
