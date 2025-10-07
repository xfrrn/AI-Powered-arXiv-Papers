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
    keyword: Optional[str] = Field(None, description="用户输入的搜索关键词")
    semantic_matching_enabled: bool = Field(False, description="是否启用了语义匹配")
    matching_method: str = Field("none", description="使用的语义匹配方法: none, tf-idf, specter-v2")
    similarity_threshold: Optional[float] = Field(None, description="语义相似度阈值")
    start_date: Optional[str] = Field(None, description="查询开始日期")
    end_date: Optional[str] = Field(None, description="查询结束日期")
    max_results: int
    time_range_start: str
    time_range_end: str
    total_fetched: int
    filtered_by_semantic: int = Field(0, description="语义匹配后的论文数量")
    summarization_enabled: bool
    categories: List[str] = Field(description="使用的 arXiv 分类列表")

class PapersResponse(BaseModel):
    """定义最终 API 响应的数据模型"""
    query_details: QueryDetails
    papers_by_field: dict[str, List[Paper]]
    papers_by_team: dict[str, List[Paper]]