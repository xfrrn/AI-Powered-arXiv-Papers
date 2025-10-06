# /arxiv_fetcher/utils.py

from datetime import datetime, timedelta, timezone

def get_last_n_days_arxiv_time_range(n_days: int = 7) -> tuple[datetime, datetime]:
    """
    获取过去 n_days 天的 arXiv 时间区间。
    每天时间区间为 UTC 19:00 到次日 19:00。
    :param n_days: 回溯天数
    :return: (start_time, end_time) datetime 对象
    """
    current_time = datetime.now(timezone.utc)
    
    end_time = current_time.replace(hour=19, minute=0, second=0, microsecond=0)
    start_time = (current_time - timedelta(days=n_days)).replace(hour=19, minute=0, second=0, microsecond=0)

    return start_time, end_time