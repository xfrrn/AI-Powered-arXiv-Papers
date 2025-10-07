# /arxiv_fetcher/models.py

from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from datetime import datetime

class Paper(BaseModel):
    """定义单篇论文的数据模型"""
    title: str
    authors: List[str]
    published: datetime
    url: str  # 改为字符串类型，简化处理
    summary: str
    primary_category: str
    summary_zh: Optional[str] = Field(None, description="由 AI 生成的中文总结")

class QueryDetails(BaseModel):
    """定义查询参数和结果的元数据"""
    start_date: Optional[str] = Field(None, description="查询开始日期")
    end_date: Optional[str] = Field(None, description="查询结束日期")
    max_results: int
    time_range_start: str
    time_range_end: str
    total_fetched: int
    summarization_enabled: bool
    categories: List[str] = Field(description="使用的 arXiv 分类列表")

class PapersResponse(BaseModel):
    """定义最终 API 响应的数据模型"""
    query_details: QueryDetails
    papers_by_field: dict[str, List[Paper]]
    papers_by_team: dict[str, List[Paper]]