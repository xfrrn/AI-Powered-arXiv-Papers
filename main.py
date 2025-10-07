from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
from datetime import datetime, date

# 导入所有需要的模块
from arxiv_fetcher.processor import get_papers_from_arxiv, process_and_classify_papers
from arxiv_fetcher.config import MAX_PAPERS, DEFAULT_CATEGORIES, SUPPORTED_CATEGORIES
from arxiv_fetcher.models import PapersResponse, QueryDetails
from arxiv_fetcher.utils import get_last_n_days_arxiv_time_range, parse_date_range
from arxiv_fetcher.semantic_matcher import semantic_filter_papers, specter_filter_papers, SENTENCE_TRANSFORMERS_AVAILABLE 

app = FastAPI(
    title="arXiv Semantic Papers API",
    description="基于 SPECTER-v2 的智能 arXiv 论文检索 API，默认使用深度学习语义匹配和 AI 论文总结。先按分类获取论文，再通过高精度语义相似度筛选最相关的内容。",
    version="2.1.0",
)

@app.get("/categories", summary="获取支持的 arXiv 分类列表")
async def get_supported_categories():
    """获取系统支持的所有 arXiv 分类及其描述"""
    return {
        "supported_categories": SUPPORTED_CATEGORIES,
        "default_categories": DEFAULT_CATEGORIES,
        "description": "使用 categories 参数时，请从 supported_categories 中选择，多个分类用逗号分隔"
    }

@app.get("/papers", response_model=PapersResponse, summary="基于分类和语义匹配的智能 arXiv 论文检索")
async def fetch_papers_api(
    keyword: Optional[str] = Query(None, description="搜索关键词，将使用语义匹配在获得的论文中进行筛选"),
    start_date: Optional[str] = Query(None, description="开始日期 (格式: YYYY-MM-DD)，不指定则使用最近1天"),
    end_date: Optional[str] = Query(None, description="结束日期 (格式: YYYY-MM-DD)，不指定则使用当前日期"),
    categories: Optional[str] = Query(None, description="arXiv 分类列表，用逗号分隔。不指定则使用默认分类"),
    max_results: int = Query(MAX_PAPERS, description="最大返回论文数量 (1-1000)", ge=1, le=1000),
    similarity_threshold: float = Query(0.75, description="语义相似度阈值 (0.0-1.0)，仅在提供关键词时有效", ge=0.0, le=1.0),
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

        # 3. 获取原始论文列表 (基于分类)
        papers = get_papers_from_arxiv(
            start_time=start_time,
            end_time=end_time,
            keywords=[],  # 不在arXiv查询中使用关键词
            categories=search_categories,
            max_results=max_results
        )

        original_count = len(papers)
        
        # 4. 如果提供了关键词，使用语义匹配进行筛选（支持 SPECTER-v2）
        semantic_matching_enabled = False
        filtered_count = original_count
        matching_method = "none"
        
        if keyword and keyword.strip():
            semantic_matching_enabled = True
            clean_keyword = keyword.strip()
            print(f"开始语义匹配，关键词: {clean_keyword}")
            print(f"相似度阈值: {similarity_threshold}")
            print(f"原始论文数量: {original_count}")
            
            # 默认使用 SPECTER-v2 进行语义匹配
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                print("使用 SPECTER-v2 进行高级语义匹配...")
                matching_method = "specter-v2"
                papers = specter_filter_papers(
                    papers=papers,
                    query=clean_keyword,
                    similarity_threshold=similarity_threshold,
                    model_name='allenai/specter2_base'
                )
            else:
                print("SPECTER-v2 不可用，回退到 TF-IDF 匹配...")
                matching_method = "tf-idf"
                papers = semantic_filter_papers(
                    papers=papers,
                    query_keyword=clean_keyword,
                    similarity_threshold=similarity_threshold,
                    use_index=True  # 启用索引优化
                )
            
            filtered_count = len(papers)
            print(f"语义匹配完成，从 {original_count} 篇筛选到 {filtered_count} 篇")

        # 5. 异步处理总结和分类
        papers_by_field, papers_by_team = await process_and_classify_papers(papers, summarize)
        
        # 6. 构建响应数据
        query_details = QueryDetails(
            keyword=keyword,
            semantic_matching_enabled=semantic_matching_enabled,
            matching_method=matching_method,
            similarity_threshold=similarity_threshold if semantic_matching_enabled else None,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            time_range_start=start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            time_range_end=end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            total_fetched=original_count,
            filtered_by_semantic=filtered_count,
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