import arxiv
from typing import List, Dict
from datetime import datetime

from .config import FIELD_CATEGORY_KEYWORDS, TEAM_CATEGORY_KEYWORDS
from .models import Paper
from .summarizer import summarize_papers_concurrently

def get_papers_from_arxiv(start_time: datetime, end_time: datetime, keywords: List[str], max_results: int) -> List[Paper]:
    """从 arXiv API 获取论文，支持多个关键词进行 OR 查询。"""
    if not keywords:
        return []

    try:
        client = arxiv.Client()
        start_time_str = start_time.strftime("%Y%m%d%H%M%S")
        end_time_str = end_time.strftime("%Y%m%d%H%M%S")
        
        # --- 核心改动：构建 OR 查询 ---
        # 将每个关键词包装成 "all:<keyword>" 的形式
        query_parts = [f'all:"{k.strip()}"' for k in keywords]
        # 用 " OR " 连接它们，并用括号括起来
        keyword_query = f"({ ' OR '.join(query_parts) })"
        
        # 最终查询语句
        query = f"{keyword_query} AND submittedDate:[{start_time_str} TO {end_time_str}]"
        
        print(f"Executing arXiv query: {query}") # 打印查询语句，方便调试

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = list(client.results(search))
        
        papers = [
            Paper(
                title=result.title,
                authors=[str(a.name) for a in result.authors],
                published=result.published,
                url=result.entry_id,
                summary=result.summary,
                primary_category=result.primary_category
            ) for result in results
        ]
        return papers
        
    except Exception as e:
        print(f"Error fetching papers from arXiv: {e}")
        return []

async def process_and_classify_papers(papers: List[Paper], enable_summary: bool) -> tuple[Dict[str, List[Paper]], Dict[str, List[Paper]]]:
    # ... (此函数保持不变)
    if enable_summary and papers:
        await summarize_papers_concurrently(papers)

    papers_by_field = {}
    for field, keywords in FIELD_CATEGORY_KEYWORDS.items():
        filtered_papers = [
            p for p in papers
            if any(k.lower() in p.summary.lower() or k.lower() in p.title.lower() for k in keywords)
        ]
        if filtered_papers:
            papers_by_field[field] = filtered_papers

    papers_by_team = {}
    for team, members in TEAM_CATEGORY_KEYWORDS.items():
        filtered_papers = [
            p for p in papers
            if any(any(m.lower() in author.lower() for author in p.authors) for m in members)
        ]
        if filtered_papers:
            papers_by_team[team] = filtered_papers
            
    return papers_by_field, papers_by_team