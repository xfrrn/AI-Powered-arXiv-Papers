from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List

# 导入所有需要的模块
from arxiv_fetcher.processor import get_papers_from_arxiv, process_and_classify_papers
from arxiv_fetcher.config import DEFAULT_KEYWORD, MAX_PAPERS
from arxiv_fetcher.models import PapersResponse, QueryDetails
from arxiv_fetcher.utils import get_last_n_days_arxiv_time_range
from arxiv_fetcher.summarizer import expand_search_keywords 

app = FastAPI(
    title="arXiv Weekly Papers API",
    description="一个集成了 AI 关键词扩展和论文总结功能的 arXiv 智能检索 API。",
    version="1.2.0",
)

@app.get("/papers", response_model=PapersResponse, summary="获取、总结并分类 arXiv 论文")
async def fetch_papers_api(
    keyword: str = Query(DEFAULT_KEYWORD, description="搜索的核心主题词"),
    days: int = Query(7, description="查询过去的天数 (1-30)", ge=1, le=30),
    max_results: int = Query(MAX_PAPERS, description="最大返回论文数量 (1-1000)", ge=1, le=1000),
    summarize: bool = Query(True, description="是否调用 AI 对论文进行总结"),
    expand_keyword: bool = Query(True, description="是否调用 AI 扩展关键词以提升搜索效果") 
):
    try:
        # 1. (可选) 扩展关键词
        search_keywords: List[str]
        if expand_keyword:
            search_keywords = await expand_search_keywords(keyword)
        else:
            search_keywords = [keyword]

        # 2. 计算时间范围
        start_time, end_time = get_last_n_days_arxiv_time_range(days)

        # 3. 获取原始论文列表 (使用关键词列表)
        papers = get_papers_from_arxiv(
            start_time=start_time,
            end_time=end_time,
            keywords=search_keywords, # <--- 传入关键词列表
            max_results=max_results
        )

        # 4. 异步处理总结和分类
        papers_by_field, papers_by_team = await process_and_classify_papers(papers, summarize)
        
        # 5. 构建响应数据
        query_details = QueryDetails(
            keyword=keyword,
            days=days,
            max_results=max_results,
            time_range_start=start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            time_range_end=end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            total_fetched=len(papers),
            summarization_enabled=summarize,
            keyword_expansion_enabled=expand_keyword,
            expanded_keywords=search_keywords if expand_keyword else None
        )
        
        return PapersResponse(
            query_details=query_details,
            papers_by_field=papers_by_field,
            papers_by_team=papers_by_team
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("服务器正在启动，请访问 http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)