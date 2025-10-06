# 🚀 AI-Powered arXiv Papers API

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于 FastAPI 的智能 arXiv 论文检索 API，集成了 AI 驱动的**关键词扩展**和**论文摘要总结**功能，旨在帮助用户更高效地追踪和理解最新的科研动态。

## ✨ 功能特点

-   **动态论文获取**: 根据英文关键词和时间范围，从 arXiv 实时检索最新论文。
-   **智能关键词扩展**: (可选) 利用 AI 模型（DeepSeek）分析核心主题词，自动生成相关领域的关键词，扩大搜索范围，提升召回率。
-   **AI 论文总结**: (可选) 自动将获取的英文论文标题和摘要，通过 AI 模型生成简洁易懂的中文核心内容总结。
-   **自定义分类**: 可在 `config.py` 文件中轻松定义研究领域和团队，API 会自动对检索结果进行筛选和分类。
-   **异步并发**: 对 AI 模型的调用采用异步并发处理，即使在总结多篇论文时也能保证较高的响应速度。
-   **交互式文档**: 基于 FastAPI 自动生成 Swagger UI 和 ReDoc 交互式 API 文档，方便调试和使用。

## 🛠️ 项目结构

```
/arxiv_fastapi_project
|
|-- .env                 # 存放 API 密钥
|-- main.py              # FastAPI 应用主文件
|-- requirements.txt     # 项目依赖
|
|-- arxiv_fetcher/
|   |-- __init__.py
|   |-- config.py        # 核心配置文件 (分类规则)
|   |-- models.py        # Pydantic 数据模型
|   |-- processor.py     # 论文获取与分类逻辑
|   |-- summarizer.py    # AI 关键词扩展与总结模块
|   |-- utils.py         # 工具函数 (如时间计算)
```

## 📚 安装与部署

### 1. 克隆项目

```bash
git clone https://github.com/xfrrn/AI-Powered-arXiv-Papers.git
cd AI-Powered-arXiv-Papers
```

### 2. 安装依赖

项目使用 `pip` 管理依赖。建议在虚拟环境中安装。

```bash
# 创建并激活虚拟环境 (可选但推荐)
python -m venv venv
source venv/bin/activate  # on Windows, use `venv\Scripts\activate`

# 安装所有依赖
pip install -r requirements.txt
```

### 3. 配置 API 密钥

本项目需要调用 DeepSeek API 来实现 AI 功能。

-   将根目录下的 `.env.example` 文件（如果提供）重命名为 `.env`。
-   在 `.env` 文件中填入你的 DeepSeek API Key：

```env
# .env
DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### 4. 启动服务

使用 Uvicorn 启动 FastAPI 应用：

```bash
uvicorn main:app --reload
```

服务启动后，你将在终端看到类似信息：
`INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

## 🚀 API 使用指南

服务启动后，打开浏览器访问 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** 即可进入交互式 API 文档 (Swagger UI)。

### 主接口: `GET /papers`

#### 参数说明

| 参数             | 类型    | 默认值      | 描述                                                       |
| ---------------- | ------- | ----------- | ---------------------------------------------------------- |
| `keyword`        | `string`| `"quantum"` | **必需**。搜索的核心主题词，**强烈建议使用英文**。               |
| `days`           | `integer`| `7`         | 查询过去的天数，范围为 1-30。                              |
| `max_results`    | `integer`| `400`       | 从 arXiv 获取的论文最大数量，范围为 1-1000。                 |
| `expand_keyword` | `boolean`| `true`      | 是否启用 AI 关键词扩展功能，以获得更全面的搜索结果。       |
| `summarize`      | `boolean`| `true`      | 是否启用 AI 论文总结功能，为每篇论文生成中文摘要。         |

#### 示例请求

获取过去 3 天内，关于 "graphene"（石墨烯）的论文，并同时启用关键词扩展和中文总结：

```
[http://127.0.0.1:8000/papers/?keyword=graphene&days=3&max_results=50&expand_keyword=true&summarize=true](http://127.0.0.1:8000/papers/?keyword=graphene&days=3&max_results=50&expand_keyword=true&summarize=true)
```

#### 响应结构

API 会返回一个 JSON 对象，包含查询详情和按领域、团队分类的论文列表。

```json
{
  "query_details": {
    "keyword": "graphene",
    "days": 3,
    "total_fetched": 25,
    "keyword_expansion_enabled": true,
    "expanded_keywords": ["graphene", "2D materials", "carbon nanotubes", ...],
    // ...
  },
  "papers_by_field": {
    "Some Category": [
      {
        "title": "A paper on Graphene...",
        "summary": "The original English abstract...",
        "summary_zh": "这是由AI生成的中文总结...",
        // ...
      }
    ]
  },
  "papers_by_team": {}
}
```

## ⚙️ 自定义配置

你可以通过修改 `/arxiv_fetcher/config.py` 文件来定制论文的分类规则。

-   **`FIELD_CATEGORY_KEYWORDS`**: 定义不同研究领域的关键词。API 会根据这些词筛选 `total_fetched` 的论文。
-   **`TEAM_CATEGORY_KEYWORDS`**: 定义不同团队或研究人员的姓名关键词，用于筛选特定作者的论文。

## 📄 开源许可

本项目采用 [MIT License](LICENSE) 开源。