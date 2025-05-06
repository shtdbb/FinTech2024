# FinTech2024：文档问答投研 Agent 系统

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)

2024 年第八届招商银行数字/科技金融训练营 FinTech 线上复赛方案

**比赛题目以及流程**详见我的小红书笔记：http://xhslink.com/a/yW6Gf8CKJ0Sbb

一个基于大语言模型的智能文档问答系统，专为金融投研和文档分析设计，支持多种文档格式和多路召回的问答机制。

## 比赛方案

`docs` 文件夹内为方案设计文档的 PDF 文件。

由于文档第四章涉及到数据集信息，为防止侵权，已删除真实数据集信息。

> 方案设计仅供参考，最终以主办方赛题为准

## 项目介绍

> 项目介绍由 Claude 3.7 自动生成，如有疑问或错漏，请提交 issue。

本项目是一个金融文档智能问答系统，可以处理各种文档格式（PDF、TXT、Markdown、Excel、CSV等），并基于多路召回和多模型协作的方式，实现高质量的文档问答、信息抽取和数据可视化功能。

### 核心特性

- **多格式文档支持**：支持 PDF、TXT、Markdown、Excel 和 CSV 等多种文档格式
- **智能分段与索引**：自动将文档分段并建立索引，支持长文档和短文档不同粒度的处理
- **多路召回机制**：结合向量检索和 BM25 等多种召回方式，提高问答相关性
- **多种问答模式**：支持普通问答、结构化信息抽取（JSON 格式）和数据可视化（图表生成）
- **上下文引用**：回答问题时提供原文引用，增强可信度和可追溯性
- **智能文档选择**：根据问题自动选择最相关的文档或文档集合
- **表格智能问答**：针对表格数据提供专门的问答能力

## 技术方案

本项目采用以下技术方案：

### 文档处理流程

1. **文档解析**：将不同格式的文档转换为统一的文本格式
2. **文本分段**：按长短不同粒度进行文本分段
3. **特征提取**：使用 Embedding 模型提取文本特征
4. **索引建立**：建立向量索引(Faiss)和 BM25 索引
5. **多路召回**：支持基于语义和关键词的混合检索策略

### 问答流程

1. **意图识别**：识别用户问题类型（普通问答、信息抽取、图表生成）
2. **文档选择**：根据问题自动选择相关文档
3. **知识检索**：从文档中检索相关内容
4. **答案生成**：基于检索内容生成答案
5. **引用标注**：为回答添加原文引用

## 项目结构

```
├── docs/                   # 项目文档
│   └── FinTech2024.pdf     # 技术方案文档
├── src/                    # 源代码
│   ├── api_server.py       # API 服务器
│   ├── chat_page.py        # 主界面和交互逻辑
│   ├── document.py         # 文档处理模块
│   ├── embedding.py        # 文本向量化模块
│   ├── Retriever.py        # 检索模块
│   ├── VectorDatabase.py   # 向量数据库
│   ├── template.py         # 提示词模板
│   ├── functions.py        # 功能函数
│   ├── logger.py           # 日志模块
│   ├── pdf2txt.py          # PDF 转文本工具
│   ├── txt2md.py           # 文本转 Markdown 工具
│   ├── utils.py            # 工具函数
│   └── finetune/           # 模型微调相关
│       ├── embedding/      # 向量模型微调
│       └── llm/            # 大语言模型微调
├── requirements.txt        # 依赖项
└── LICENSE.txt             # 许可证
```

## 主要组件介绍

### 1. 文档处理模块 (document.py)

处理各种格式的文档，包括文本文档、PDF文档和表格文档，提供文档解析、分段和管理功能。

- `TextDocument`: 处理文本类文档
- `PDFDocument`: 处理PDF文档
- `SheetDocument`: 处理表格类文档

### 2. 向量化和检索模块 (embedding.py, Retriever.py)

负责文本向量化和多路召回机制：

- `EmbeddingAPI`: 文本向量化接口
- `BM25`: 基于 BM25 算法的文本检索
- `Route`: 定义检索路由策略
- `Retriever`: 多路召回检索器
- `HyDE`: 基于假设的检索增强

### 3. 用户界面和交互模块 (chat_page.py)

基于 Streamlit 构建的用户界面，处理用户上传文档、问答交互等功能。

- 文档上传和处理
- 问答交互
- 结果呈现和引用标注

### 4. 问答模板 (template.py)

包含各种问答场景的提示词模板：

- 普通问答模板
- 结构化信息抽取模板
- 图表生成模板

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动系统

```bash
cd src
streamlit run chat_page.py
```

### 使用流程

1. 通过 Web 界面上传文档
2. 等待系统处理文档（分段、向量化、索引建立）
3. 在对话框中输入问题
4. 获取带有原文引用的回答

## 高级功能

- **信息抽取**：可以从文档中抽取结构化信息，以 JSON 格式返回
- **数据可视化**：自动生成数据图表，支持折线图、柱状图等
- **多文档比较**：可以比较多个文档中的信息
- **表格智能问答**：针对表格数据进行智能问答和分析

## 致谢

本项目中 PDF 转 TXT 工具使用了来源于 [FinGLM](https://github.com/MetaGLM/FinGLM) 的代码。感谢 FinGLM 项目团队的贡献。

由于 FinGLM 项目未设置明确的开源协议，使用时请遵循其公益性质和相关规定。

## 引用

您可以按照以下 BibTex 格式引用本仓库：
```bibtex
@software{FinTech2024,
  author = {shtdbb},
  title = {FinTech2024：文档问答投研 Agent 系统},
  url = {https://github.com/shtdbb/FinTech2024},
  year = {2025}
}
```

## 许可证

本项目采用 MIT 许可证，详情请参阅 [LICENSE.txt](LICENSE.txt) 文件。
