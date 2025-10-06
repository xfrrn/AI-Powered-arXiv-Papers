import os
import asyncio
import json
from typing import List
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .models import Paper

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化 DeepSeek 的异步客户端
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# --- 新增函数 ---
async def expand_search_keywords(base_keyword: str) -> List[str]:
    """
    调用 DeepSeek API 扩展用户输入的初始关键词。
    返回一个包含原始关键词和扩展关键词的列表。
    """
    if not base_keyword:
        return []

    prompt = f"""
    你是一个科研领域的专家，特别擅长构建精确的 arXiv 检索词。
    请根据用户提供的核心主题词 "{base_keyword}"，生成 5 个最相关的、用于在 arXiv 上搜索的英文关键词或短语。

    要求：
    1.  生成的关键词必须是该领域的核心术语或高度相关的技术。
    2.  返回一个 JSON 格式的数组，其中只包含字符串。
    3.  不要包含任何解释或多余的文本，直接返回 JSON 数组。

    例如，如果输入 "quantum computing"，一个好的输出是：
    ["quantum algorithms", "qubit", "quantum supremacy", "quantum simulation", "quantum entanglement"]
    """

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个科研领域的专家，专门生成 arXiv 检索词。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.6,
            response_format={"type": "json_object"}, # 请求 JSON 输出
        )
        if response.choices:
            content = response.choices[0].message.content
            # 解析 JSON 字符串
            expanded_list = json.loads(content)
            # 确保返回的是一个字符串列表，并将原始关键词加在最前面
            if isinstance(expanded_list, list) and all(isinstance(item, str) for item in expanded_list):
                return [base_keyword] + expanded_list
    except Exception as e:
        print(f"Keyword expansion failed for '{base_keyword}': {e}")
    
    # 如果扩展失败，则只返回原始关键词
    return [base_keyword]


async def summarize_paper(paper: Paper) -> None:
    # ... (此函数保持不变)
    if not paper.title or not paper.summary:
        return

    prompt = f"""
    你是一个专业的科研助理，擅长将复杂的英文学术论文摘要翻译并提炼成通俗易懂的中文核心内容。
    请根据以下论文的标题和摘要，用中文总结出这篇论文的核心观点和主要贡献，字数控制在100字以内。

    - 标题 (Title): {paper.title}
    - 摘要 (Abstract): {paper.summary}

    中文总结：
    """
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的科研助理。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.5,
        )
        if response.choices:
            paper.summary_zh = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summarization failed for paper '{paper.title}': {e}")
        paper.summary_zh = f"总结失败: {str(e)}"

async def summarize_papers_concurrently(papers: List[Paper]) -> None:
    # ... (此函数保持不变)
    tasks = [summarize_paper(paper) for paper in papers]
    await asyncio.gather(*tasks)