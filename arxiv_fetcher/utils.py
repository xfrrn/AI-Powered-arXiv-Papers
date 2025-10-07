# /arxiv_fetcher/utils.py

from datetime import datetime, timedelta, timezone, date
from typing import Tuple, Optional

def get_last_n_days_arxiv_time_range(n_days: int = 7) -> tuple[datetime, datetime]:
    """
    获取过去 n_days 天的 arXiv 时间区间。
    每天时间区间为 UTC 00:00 到次日 00:00。
    :param n_days: 回溯天数
    :return: (start_time, end_time) datetime 对象
    """
    current_time = datetime.now(timezone.utc)
    
    end_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = (current_time - timedelta(days=n_days)).replace(hour=0, minute=0, second=0, microsecond=0)

    return start_time, end_time

def parse_date_range(start_date: Optional[str], end_date: Optional[str]) -> Tuple[datetime, datetime]:
    """
    解析用户提供的日期范围，转换为 arXiv 查询使用的 datetime 对象。
    如果没有提供日期，则使用默认的最近1天。
    
    :param start_date: 开始日期字符串 (格式: YYYY-MM-DD)
    :param end_date: 结束日期字符串 (格式: YYYY-MM-DD)
    :return: (start_time, end_time) datetime 对象
    """
    if not start_date and not end_date:
        # 如果都没有提供，使用默认的最近1天
        return get_last_n_days_arxiv_time_range(1)
    
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
        else:
            # 如果只提供了结束日期，开始日期设为1天前
            if end_date:
                end_dt_temp = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                start_dt = (end_dt_temp - timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            else:
                # 不应该到达这里，因为前面已经检查过
                return get_last_n_days_arxiv_time_range(1)
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
        else:
            # 如果只提供了开始日期，结束日期设为当前时间
            end_dt = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        
        # 确保开始日期不晚于结束日期
        if start_dt > end_dt:
            raise ValueError("开始日期不能晚于结束日期")
        
        return start_dt, end_dt
        
    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("日期格式错误，请使用 YYYY-MM-DD 格式")
        else:
            raise e