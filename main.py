from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
from datetime import datetime, date

# 导入所有需要的模块
from arxiv_fetcher.processor import get_papers_from_arxiv, process_and_classify_papers
from arxiv_fetcher.config import MAX_PAPERS, DEFAULT_CATEGORIES, SUPPORTED_CATEGORIES
from arxiv_fetcher.models import PapersResponse, QueryDetails
from arxiv_fetcher.utils import get_last_n_days_arxiv_time_range, parse_date_range
from arxiv_fetcher.summarizer import expand_search_keywords 

app = FastAPI(
    title="arXiv Weekly Papers API",
    description="一个集成了 AI 关键词扩展和论文总结功能的 arXiv 智能检索 API，支持按分类和关键词组合查询。",
    version="1.3.0",
)

@app.get("/categories", summary="获取支持的 arXiv 分类列表")
async def get_supported_categories():
    """获取系统支持的所有 arXiv 分类及其描述"""
    return {
        "supported_categories": SUPPORTED_CATEGORIES,
        "default_categories": DEFAULT_CATEGORIES,
        "description": "使用 categories 参数时，请从 supported_categories 中选择，多个分类用逗号分隔"
    }

@app.get("/papers", response_model=PapersResponse, summary="基于配置的分类获取 arXiv 论文")
async def fetch_papers_api(
    start_date: Optional[str] = Query(None, description="开始日期 (格式: YYYY-MM-DD)，不指定则使用最近1天"),
    end_date: Optional[str] = Query(None, description="结束日期 (格式: YYYY-MM-DD)，不指定则使用当前日期"),
    categories: Optional[str] = Query(None, description="arXiv 分类列表，用逗号分隔。不指定则使用默认分类"),
    max_results: int = Query(MAX_PAPERS, description="最大返回论文数量 (1-1000)", ge=1, le=1000),
    summarize: bool = Query(True, description="是否调用 AI 对论文进行总结")
):
    try:
        # 1. 处理分类参数
        search_categories: List[str]
        if categories:
            # 将逗号分隔的字符串转换为列表
            search_categories = [cat.strip() for cat in categories.split(',') if cat.strip()]
            # 验证分类是否支持
            invalid_categories = [cat for cat in search_categories if cat not in SUPPORTED_CATEGORIES]
            if invalid_categories:
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的分类: {', '.join(invalid_categories)}. 支持的分类: {', '.join(SUPPORTED_CATEGORIES.keys())}"
                )
        else:
            # 如果没有指定分类，使用默认分类
            search_categories = DEFAULT_CATEGORIES

        # 2. 解析日期范围
        start_time, end_time = parse_date_range(start_date, end_date)

        # 3. 获取原始论文列表 (仅使用分类，不使用关键词)
        papers = get_papers_from_arxiv(
            start_time=start_time,
            end_time=end_time,
            keywords=[],  # 不使用关键词筛选
            categories=search_categories,
            max_results=max_results
        )

        # 4. 异步处理总结和分类
        papers_by_field, papers_by_team = await process_and_classify_papers(papers, summarize)
        
        # 5. 构建响应数据
        query_details = QueryDetails(
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            time_range_start=start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            time_range_end=end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            total_fetched=len(papers),
            summarization_enabled=summarize,
            categories=search_categories
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