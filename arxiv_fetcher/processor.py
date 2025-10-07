import arxiv
from typing import List, Dict, Optional
from datetime import datetime

from .models import Paper
from .summarizer import summarize_papers_concurrently

def get_papers_from_arxiv(
    start_time: datetime, 
    end_time: datetime, 
    keywords: List[str], 
    categories: Optional[List[str]] = None,
    max_results: int = 300
) -> List[Paper]:
    """
    从 arXiv API 获取论文，主要基于分类进行查询。
    简化后的逻辑：优先使用分类，关键词为辅助（可选）。
    """
    try:
        client = arxiv.Client()
        start_time_str = start_time.strftime("%Y%m%d%H%M%S")
        end_time_str = end_time.strftime("%Y%m%d%H%M%S")
        
        print("正在构建基于分类的查询...")
        
        # 用于存放各个查询部分
        query_parts = []
        
        # 1. 日期部分 (必须)
        date_query = f"submittedDate:[{start_time_str} TO {end_time_str}]"
        query_parts.append(date_query)

        # 2. 分类部分 (主要筛选条件)
        if categories and len(categories) > 0:
            category_query = " OR ".join([f'cat:{c}' for c in categories])
            query_parts.append(f"({category_query})")
            print(f"使用分类: {', '.join(categories)}")
        else:
            print("警告：没有提供分类，将搜索所有论文")

        # 3. 关键词部分 (可选的辅助筛选)
        if keywords and len(keywords) > 0:
            # 使用 OR 连接，表示只要匹配任意一个关键词即可
            keyword_query = " OR ".join([f'(ti:"{k.strip()}" OR abs:"{k.strip()}")' for k in keywords])
            query_parts.append(f"({keyword_query})")
            print(f"附加关键词筛选: {', '.join(keywords)}")
            
        # 构建最终查询
        if len(query_parts) == 1:
            # 只有日期条件
            final_query = date_query
        else:
            # 使用 AND 将所有部分连接成最终查询
            final_query = " AND ".join(query_parts)
        
        print(f"最终生成的查询语句: {final_query}")

        search = arxiv.Search(
            query=final_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        print("正在从 arXiv API 获取数据，请稍候...")
        results = list(client.results(search))
        
        print(f"搜索完成，共找到 {len(results)} 篇论文。")
        
        papers = [
            Paper(
                title=result.title,
                authors=[str(a.name) for a in result.authors],
                published=result.published,
                url=result.entry_id,  # 现在是字符串类型
                summary=result.summary,
                primary_category=result.primary_category,
                summary_zh=None  # AI 总结将在后续步骤中添加
            ) for result in results
        ]
        return papers
        
    except Exception as e:
        print(f"Error fetching papers from arXiv: {e}")
        return []

async def process_and_classify_papers(papers: List[Paper], enable_summary: bool) -> tuple[Dict[str, List[Paper]], Dict[str, List[Paper]]]:
    """
    处理论文：生成摘要总结。
    简化版本：不再进行复杂的领域和团队分类，直接按 arXiv 分类分组。
    """
    # 1. 生成 AI 总结
    if enable_summary and papers:
        await summarize_papers_concurrently(papers)

    # 2. 按 arXiv 分类分组论文
    papers_by_field = {}
    for paper in papers:
        category = paper.primary_category
        if category not in papers_by_field:
            papers_by_field[category] = []
        papers_by_field[category].append(paper)

    # 3. 不再按团队分类，返回空字典
    papers_by_team = {}
            
    return papers_by_field, papers_by_team