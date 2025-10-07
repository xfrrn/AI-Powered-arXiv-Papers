"""
语义匹配模块
使用语义相似度进行论文匹配的简化实现
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from scipy.sparse import spmatrix
import logging
import re
from .models import Paper

# 尝试导入深度学习相关依赖
try:
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    TORCH_AVAILABLE = True
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    TORCH_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSemanticMatcher:
    """优化的语义匹配器，支持索引构建和快速查询"""
    
    def __init__(self, similarity_threshold: float = 0.1):
        """
        初始化语义匹配器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值的论文会被保留
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
        
        # 索引相关的属性
        self.papers_index: Optional[List[Paper]] = None  # 存储论文对象的索引
        self.papers_vectors: Optional[Union[spmatrix, np.ndarray, List[str]]] = None  # 存储论文的TF-IDF向量
        self.is_indexed: bool = False  # 标记是否已构建索引
        
        self._init_vectorizer()
    
    def _init_vectorizer(self):
        """初始化文本向量化工具"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            
            # 创建TF-IDF向量化器
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)  # 使用1-gram和2-gram
            )
            self.cosine_similarity = cos_sim
            logger.info("TF-IDF向量化器初始化成功")
            
        except ImportError as e:
            logger.error(f"scikit-learn未安装: {e}")
            self.vectorizer = None
            self.cosine_similarity = None
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 转换为小写
        text = text.lower()
        # 移除特殊字符，保留字母数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def _simple_word_overlap(self, query: str, text: str) -> float:
        """简单的词汇重叠相似度计算"""
        query_words = set(self._preprocess_text(query).split())
        text_words = set(self._preprocess_text(text).split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        # Jaccard相似度
        return len(intersection) / len(union) if union else 0.0
    
    def build_index(self, papers: List[Paper]) -> bool:
        """
        构建论文的TF-IDF向量索引
        
        Args:
            papers: 论文列表
            
        Returns:
            bool: 索引构建是否成功
        """
        if not papers:
            logger.warning("论文列表为空，无法构建索引")
            return False
        
        try:
            logger.info(f"开始构建 {len(papers)} 篇论文的语义索引...")
            
            # 准备论文文本
            paper_texts = []
            for paper in papers:
                combined_text = f"{paper.title} {paper.summary}"
                paper_texts.append(self._preprocess_text(combined_text))
            
            if self.vectorizer is not None:
                # 使用TF-IDF构建索引
                logger.info("使用TF-IDF方法构建索引")
                
                # 训练向量化器并转换论文文本
                self.papers_vectors = self.vectorizer.fit_transform(paper_texts)
                
                # 存储论文索引
                self.papers_index = papers.copy()
                self.is_indexed = True
                
                logger.info(f"TF-IDF索引构建完成，向量维度: {self.papers_vectors.shape}")
                return True
            else:
                # 回退到简单匹配，直接存储论文和预处理文本
                logger.info("使用简单匹配方法构建索引")
                self.papers_index = papers.copy()
                self.papers_vectors = paper_texts  # 存储预处理后的文本
                self.is_indexed = True
                return True
                
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            self.is_indexed = False
            return False
    
    def clear_index(self):
        """清除已构建的索引"""
        self.papers_index = None
        self.papers_vectors = None
        self.is_indexed = False
        logger.info("索引已清除")
    
    def filter_papers_by_semantic_similarity(
        self, 
        papers: List[Paper], 
        query_keyword: str,
        top_k: Optional[int] = None,
        use_index: bool = True
    ) -> List[Tuple[Paper, float]]:
        """
        根据语义相似度筛选论文（优化版本）
        
        Args:
            papers: 论文列表（如果use_index=True，则忽略此参数）
            query_keyword: 查询关键词
            top_k: 返回前k个最相似的论文，None表示返回所有超过阈值的论文
            use_index: 是否使用预构建的索引（推荐）
            
        Returns:
            按相似度排序的论文列表，每个元素是(论文, 相似度分数)的元组
        """
        if not query_keyword.strip():
            return []
        
        # 决定使用索引还是传入的论文列表
        if use_index and self.is_indexed:
            # 使用预构建的索引（快速路径）
            return self._query_with_index(query_keyword, top_k)
        elif papers:
            # 传统方法：对传入的论文列表进行实时计算（慢速路径）
            return self._query_without_index(papers, query_keyword, top_k)
        else:
            logger.warning("既没有构建索引也没有提供论文列表")
            return []
    
    def _query_with_index(self, query_keyword: str, top_k: Optional[int] = None) -> List[Tuple[Paper, float]]:
        """使用预构建索引进行快速查询"""
        if not self.is_indexed or self.papers_index is None or self.papers_vectors is None:
            logger.error("索引未构建或数据不完整，无法进行快速查询")
            return []
        
        try:
            paper_similarity_pairs: List[Tuple[Paper, float]] = []
            processed_query = self._preprocess_text(query_keyword)
            
            if (self.vectorizer is not None and self.cosine_similarity is not None and 
                hasattr(self.papers_vectors, 'shape') and 
                isinstance(self.papers_vectors, (spmatrix, np.ndarray))):
                # 使用TF-IDF + 余弦相似度（快速路径）
                logger.info("使用预构建TF-IDF索引进行快速查询")
                
                # 只对查询关键词进行转换（不需要重新训练）
                query_vector = self.vectorizer.transform([processed_query])
                
                # 计算查询向量与所有预存论文向量的相似度
                similarities = self.cosine_similarity(query_vector, self.papers_vectors).flatten()
                
                paper_similarity_pairs = [(paper, float(sim)) for paper, sim in zip(self.papers_index, similarities)]
                
            else:
                # 回退到简单词汇重叠（使用预存的预处理文本）
                logger.info("使用预构建简单匹配索引进行查询")
                if isinstance(self.papers_vectors, list):
                    # papers_vectors存储的是预处理的文本列表
                    for paper, paper_text in zip(self.papers_index, self.papers_vectors):
                        similarity = self._simple_word_overlap(processed_query, str(paper_text))
                        paper_similarity_pairs.append((paper, similarity))
                else:
                    logger.warning("无法识别预存的向量格式，回退到实时计算")
                    # 如果索引数据不可用，实时计算
                    for paper in self.papers_index:
                        combined_text = f"{paper.title} {paper.summary}"
                        similarity = self._simple_word_overlap(processed_query, combined_text)
                        paper_similarity_pairs.append((paper, similarity))
            
            # 按相似度降序排序
            paper_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 筛选结果
            if top_k is not None:
                filtered_results = paper_similarity_pairs[:top_k]
            else:
                filtered_results = [
                    (paper, sim) for paper, sim in paper_similarity_pairs 
                    if sim >= self.similarity_threshold
                ]
            
            logger.info(f"快速查询完成，从索引中的 {len(self.papers_index)} 篇论文中筛选出 {len(filtered_results)} 篇")
            if filtered_results:
                logger.info(f"最高相似度: {filtered_results[0][1]:.3f}")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"索引查询过程中出错: {e}")
            return []
    
    def _query_without_index(self, papers: List[Paper], query_keyword: str, top_k: Optional[int] = None) -> List[Tuple[Paper, float]]:
        """不使用索引的传统查询方法（向后兼容）"""
        try:
            paper_similarity_pairs = []
            
            if self.vectorizer is not None and self.cosine_similarity is not None:
                # 使用TF-IDF + 余弦相似度
                paper_texts = []
                for paper in papers:
                    combined_text = f"{paper.title} {paper.summary}"
                    paper_texts.append(self._preprocess_text(combined_text))
                
                # 添加查询关键词
                all_texts = [self._preprocess_text(query_keyword)] + paper_texts
                
                # 计算TF-IDF矩阵
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                
                # 计算查询与所有论文的相似度
                full_similarity = self.cosine_similarity(tfidf_matrix)
                similarities = full_similarity[0, 1:]
                
                paper_similarity_pairs = list(zip(papers, similarities))
                
            else:
                # 回退到简单词汇重叠
                logger.info("使用简单词汇重叠方法")
                for paper in papers:
                    combined_text = f"{paper.title} {paper.summary}"
                    similarity = self._simple_word_overlap(query_keyword, combined_text)
                    paper_similarity_pairs.append((paper, similarity))
            
            # 按相似度降序排序
            paper_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 筛选结果
            if top_k is not None:
                filtered_results = paper_similarity_pairs[:top_k]
            else:
                filtered_results = [
                    (paper, sim) for paper, sim in paper_similarity_pairs 
                    if sim >= self.similarity_threshold
                ]
            
            logger.info(f"传统查询完成，从 {len(papers)} 篇论文中筛选出 {len(filtered_results)} 篇")
            if filtered_results:
                logger.info(f"最高相似度: {filtered_results[0][1]:.3f}")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"传统查询过程中出错: {e}")
            return [(paper, 0.0) for paper in papers]

# 全局实例，避免重复初始化
_semantic_matcher_instance = None

def get_semantic_matcher() -> SimpleSemanticMatcher:
    """获取语义匹配器的单例实例"""
    global _semantic_matcher_instance
    if _semantic_matcher_instance is None:
        _semantic_matcher_instance = SimpleSemanticMatcher()
    return _semantic_matcher_instance

def semantic_filter_papers(
    papers: List[Paper], 
    query_keyword: str, 
    similarity_threshold: float = 0.1,
    top_k: Optional[int] = None,
    use_index: bool = True
) -> List[Paper]:
    """
    便捷函数：使用语义相似度筛选论文（支持索引优化）
    
    Args:
        papers: 论文列表
        query_keyword: 查询关键词
        similarity_threshold: 相似度阈值
        top_k: 返回前k个最相似的论文
        use_index: 是否构建和使用索引（推荐True以提高性能）
        
    Returns:
        筛选后的论文列表
    """
    matcher = get_semantic_matcher()
    matcher.similarity_threshold = similarity_threshold
    
    if use_index and papers:
        # 构建索引并使用快速查询
        matcher.build_index(papers)
        results = matcher.filter_papers_by_semantic_similarity([], query_keyword, top_k, use_index=True)
    else:
        # 传统方法
        results = matcher.filter_papers_by_semantic_similarity(papers, query_keyword, top_k, use_index=False)
    
    # 只返回论文对象，不返回相似度分数
    return [paper for paper, _ in results]


def create_semantic_index(papers: List[Paper], similarity_threshold: float = 0.1) -> SimpleSemanticMatcher:
    """
    便利函数：为论文列表创建语义索引
    
    Args:
        papers: 论文列表
        similarity_threshold: 相似度阈值
        
    Returns:
        已构建索引的语义匹配器实例
    """
    matcher = SimpleSemanticMatcher(similarity_threshold=similarity_threshold)
    matcher.build_index(papers)
    return matcher


def query_semantic_index(
    matcher: SimpleSemanticMatcher, 
    query_keyword: str, 
    top_k: Optional[int] = None
) -> List[Tuple[Paper, float]]:
    """
    便利函数：使用预构建的语义索引进行查询
    
    Args:
        matcher: 已构建索引的语义匹配器
        query_keyword: 查询关键词
        top_k: 返回前k个最相似的论文
        
    Returns:
        (论文, 相似度分数)的列表
    """
    if not matcher.is_indexed:
        logger.warning("匹配器未构建索引，无法进行快速查询")
        return []
    
    return matcher.filter_papers_by_semantic_similarity([], query_keyword, top_k, use_index=True)


class SpecterMatcher:
    """
    基于 SPECTER-v2 的高级语义匹配器
    使用深度学习模型进行更准确的语义理解和相似度计算
    """
    
    def __init__(self, 
                 model_name: str = 'allenai/specter2_base',
                 similarity_threshold: float = 0.75,
                 device: Optional[str] = None):
        """
        初始化 SPECTER 语义匹配器
        
        Args:
            model_name: 要使用的模型名称，默认为 SPECTER-v2
            similarity_threshold: 相似度阈值
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        # 设备选择
        if device is None:
            if TORCH_AVAILABLE and torch is not None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        # 索引相关属性
        self.papers_index: Optional[List[Paper]] = None
        self.papers_vectors = None  # 类型: Optional[torch.Tensor]
        self.is_indexed: bool = False
        
        # 模型相关
        self.model = None  # 类型: Optional[SentenceTransformer]
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化 SPECTER-v2 模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers 未安装，无法使用 SPECTER 匹配器")
            logger.error("请运行: pip install sentence-transformers torch")
            return
        
        try:
            logger.info(f"正在加载 SPECTER 模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            
                # 优化设置
                if self.device == 'cuda' and self.model is not None:
                    try:
                        self.model.half()  # 使用半精度以节省显存
                    except Exception as e:
                        logger.warning(f"无法启用半精度模式: {e}")
            else:
                logger.error("SentenceTransformer 不可用")
                
            logger.info("SPECTER 模型加载成功")
            
        except Exception as e:
            logger.error(f"加载 SPECTER 模型失败: {e}")
            logger.error("将回退到简单语义匹配")
            self.model = None
    
    def build_index(self, papers: List[Paper]) -> bool:
        """
        使用 SPECTER-v2 构建语义向量索引
        
        Args:
            papers: 论文列表
            
        Returns:
            bool: 索引构建是否成功
        """
        if not papers:
            logger.warning("论文列表为空，无法构建索引")
            return False
        
        if self.model is None:
            logger.error("SPECTER 模型未初始化，无法构建索引")
            return False
        
        try:
            logger.info(f"开始使用 SPECTER-v2 构建 {len(papers)} 篇论文的深度语义索引...")
            
            # 准备论文文本（对于 SPECTER，直接使用原始文本通常效果更好）
            corpus = []
            for paper in papers:
                # SPECTER 专门为学术论文设计，标题和摘要的组合是最佳实践
                combined_text = f"{paper.title}. {paper.summary}"
                corpus.append(combined_text)
            
            # 使用 SPECTER 模型编码（这是计算密集型操作）
            logger.info("正在进行深度语义编码，请耐心等待...")
            
            if TORCH_AVAILABLE and torch is not None:
                with torch.no_grad():  # 推理时不需要梯度，节省内存
                    self.papers_vectors = self.model.encode(
                        corpus,
                        convert_to_tensor=True,
                        show_progress_bar=True,
                        batch_size=32,  # 根据显存大小调整
                        normalize_embeddings=True  # 归一化有助于相似度计算
                    )
            else:
                self.papers_vectors = self.model.encode(
                    corpus,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                    batch_size=32,
                    normalize_embeddings=True
                )
            
            # 存储论文索引
            self.papers_index = papers.copy()
            self.is_indexed = True
            
            vector_shape = self.papers_vectors.shape
            memory_usage = self.papers_vectors.element_size() * self.papers_vectors.nelement() / (1024**2)
            
            logger.info(f"SPECTER 索引构建完成！")
            logger.info(f"向量维度: {vector_shape}")
            logger.info(f"内存使用: {memory_usage:.2f} MB")
            logger.info(f"设备: {self.papers_vectors.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"SPECTER 索引构建失败: {e}")
            self.is_indexed = False
            return False
    
    def search(self, 
               query: str, 
               top_k: Optional[int] = None,
               return_scores: bool = True) -> List[Tuple[Paper, float]]:
        """
        使用 SPECTER 索引进行高效语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回前k个最相似结果，None表示返回所有超过阈值的
            return_scores: 是否返回相似度分数
            
        Returns:
            按相似度排序的 (论文, 相似度分数) 列表
        """
        if not self.is_indexed or self.model is None or self.papers_vectors is None or self.papers_index is None:
            logger.error("索引未构建或模型未加载，无法进行搜索")
            return []
        
        if not query.strip():
            logger.warning("查询文本为空")
            return []
        
        try:
            # 编码查询文本（单个查询，速度很快）
            if TORCH_AVAILABLE and torch is not None:
                with torch.no_grad():
                    query_embedding = self.model.encode(
                        query,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
            else:
                query_embedding = self.model.encode(
                    query,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
            
            # 计算与所有论文的相似度（利用GPU并行计算）
            if SENTENCE_TRANSFORMERS_AVAILABLE and util is not None:
                similarities = util.cos_sim(query_embedding, self.papers_vectors)[0]
            else:
                logger.error("sentence-transformers util 不可用")
                return []
            
            # 根据需求确定返回数量
            if top_k is not None and TORCH_AVAILABLE and torch is not None:
                # 使用 torch.topk 高效获取前k个结果
                top_results = torch.topk(similarities, k=min(top_k, len(similarities)))
                indices = top_results.indices
                scores = top_results.values
            elif TORCH_AVAILABLE and torch is not None:
                # 获取所有超过阈值的结果
                mask = similarities >= self.similarity_threshold
                indices = torch.nonzero(mask).squeeze(-1)
                scores = similarities[indices]
                
                # 按分数排序
                sorted_indices = torch.argsort(scores, descending=True)
                indices = indices[sorted_indices]
                scores = scores[sorted_indices]
            else:
                logger.error("torch 不可用，无法处理相似度计算")
                return []
            
            # 构建结果
            results = []
            for idx, score in zip(indices, scores):
                idx_int = int(idx.item()) if hasattr(idx, 'item') else int(idx)
                paper = self.papers_index[idx_int]
                similarity_score = float(score.item()) if hasattr(score, 'item') else float(score)
                results.append((paper, similarity_score))
            
            logger.info(f"SPECTER 搜索完成，从 {len(self.papers_index)} 篇论文中找到 {len(results)} 个匹配结果")
            if results:
                logger.info(f"最高相似度: {results[0][1]:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"SPECTER 搜索过程中出错: {e}")
            return []
    
    def clear_index(self):
        """清除索引并释放GPU内存"""
        if self.papers_vectors is not None:
            del self.papers_vectors
            if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.papers_index = None
        self.papers_vectors = None
        self.is_indexed = False
        logger.info("SPECTER 索引已清除，GPU内存已释放")
    
    def get_embedding_dim(self) -> int:
        """获取模型的嵌入维度"""
        if self.model is not None:
            dim = self.model.get_sentence_embedding_dimension()
            return dim if dim is not None else 0
        return 0
    
    def is_model_loaded(self) -> bool:
        """检查模型是否成功加载"""
        return self.model is not None


# SPECTER 相关的便利函数
def create_specter_matcher(
    model_name: str = 'allenai/specter2_base',
    similarity_threshold: float = 0.75,
    device: Optional[str] = None
) -> Optional[SpecterMatcher]:
    """
    便利函数：创建 SPECTER 语义匹配器
    
    Args:
        model_name: SPECTER 模型名称
        similarity_threshold: 相似度阈值
        device: 计算设备
        
    Returns:
        SPECTER 匹配器实例，如果创建失败则返回 None
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers 未安装，无法创建 SPECTER 匹配器")
        return None
    
    try:
        matcher = SpecterMatcher(
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            device=device
        )
        
        if matcher.is_model_loaded():
            logger.info(f"SPECTER 匹配器创建成功，模型维度: {matcher.get_embedding_dim()}")
            return matcher
        else:
            logger.error("SPECTER 模型加载失败")
            return None
            
    except Exception as e:
        logger.error(f"创建 SPECTER 匹配器失败: {e}")
        return None


def specter_filter_papers(
    papers: List[Paper],
    query: str,
    model_name: str = 'allenai/specter2_base',
    similarity_threshold: float = 0.75,
    top_k: Optional[int] = None,
    device: Optional[str] = None
) -> List[Paper]:
    """
    便利函数：使用 SPECTER-v2 进行高级语义筛选
    
    Args:
        papers: 论文列表
        query: 查询文本
        model_name: SPECTER 模型名称
        similarity_threshold: 相似度阈值
        top_k: 返回前k个结果
        device: 计算设备
        
    Returns:
        筛选后的论文列表
    """
    matcher = create_specter_matcher(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        device=device
    )
    
    if matcher is None:
        logger.warning("SPECTER 匹配器创建失败，回退到简单语义匹配")
        return semantic_filter_papers(papers, query, similarity_threshold, top_k, use_index=True)
    
    # 构建索引并搜索
    if matcher.build_index(papers):
        results = matcher.search(query, top_k)
        matcher.clear_index()  # 释放内存
        return [paper for paper, _ in results]
    else:
        logger.error("SPECTER 索引构建失败")
        return []


def specter_search_with_scores(
    papers: List[Paper],
    query: str,
    model_name: str = 'allenai/specter2_base',
    similarity_threshold: float = 0.75,
    top_k: Optional[int] = None,
    device: Optional[str] = None
) -> List[Tuple[Paper, float]]:
    """
    便利函数：使用 SPECTER-v2 搜索并返回相似度分数
    
    Args:
        papers: 论文列表
        query: 查询文本
        model_name: SPECTER 模型名称
        similarity_threshold: 相似度阈值  
        top_k: 返回前k个结果
        device: 计算设备
        
    Returns:
        (论文, 相似度分数) 的列表
    """
    matcher = create_specter_matcher(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        device=device
    )
    
    if matcher is None:
        logger.warning("SPECTER 匹配器创建失败，回退到简单语义匹配")
        simple_matcher = get_semantic_matcher()
        simple_matcher.similarity_threshold = similarity_threshold
        simple_matcher.build_index(papers)
        results = simple_matcher.filter_papers_by_semantic_similarity([], query, top_k, use_index=True)
        simple_matcher.clear_index()
        return results
    
    # 构建索引并搜索
    if matcher.build_index(papers):
        results = matcher.search(query, top_k)
        matcher.clear_index()  # 释放内存
        return results
    else:
        logger.error("SPECTER 索引构建失败")
        return []