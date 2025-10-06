# /arxiv_fetcher/models.py

from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from datetime import datetime

class Paper(BaseModel):
    """定义单篇论文的数据模型"""
    title: str
    authors: List[str]
    published: datetime
    url: HttpUrl
    summary: str
    primary_category: str
    summary_zh: Optional[str] = Field(None, description="由 AI 生成的中文总结")

class QueryDetails(BaseModel):
    """定义查询参数和结果的元数据"""
    keyword: str
    days: int
    max_results: int
    time_range_start: str
    time_range_end: str
    total_fetched: int
    summarization_enabled: bool
    keyword_expansion_enabled: bool
    expanded_keywords: Optional[List[str]] = Field(None, description="由 AI 扩展后的搜索关键词列表")

class PapersResponse(BaseModel):
    """定义最终 API 响应的数据模型"""
    query_details: QueryDetails
    papers_by_field: dict[str, List[Paper]]
    papers_by_team: dict[str, List[Paper]]