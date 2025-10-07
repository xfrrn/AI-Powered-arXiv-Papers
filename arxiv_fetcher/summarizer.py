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

async def summarize_paper(paper: Paper) -> None:
    # ... (此函数保持不变)
    if not paper.title or not paper.summary:
        return

    prompt = f"""
        你是一个专业的科研助理，擅长将复杂的英文学术论文摘要翻译并提炼成通俗易懂的中文核心内容。
        请根据以下论文的标题和摘要，严格按照下面的三段式结构，用中文总结出这篇论文的核心内容，总字数控制在100-150字之间。

        - **背景与问题**: 这篇论文试图解决什么领域里的什么具体问题？
        - **核心方法**: 它提出了什么新的方法、模型或架构？
        - **主要贡献**: 最终取得了什么关键结果或做出了什么主要贡献？

        ---
        - 标题 (Title): {paper.title}
        - 摘要 (Abstract): {paper.summary}
        ---

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
            content = response.choices[0].message.content
            paper.summary_zh = content.strip() if content is not None else ""
    except Exception as e:
        print(f"Summarization failed for paper '{paper.title}': {e}")
        paper.summary_zh = f"总结失败: {str(e)}"

async def summarize_papers_concurrently(papers: List[Paper]) -> None:
    tasks = [summarize_paper(paper) for paper in papers]
    await asyncio.gather(*tasks)