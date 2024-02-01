import time 

class Timer:
    """计时器类，用于计算程序运行时间和估计剩余时间。

    Args:
        timer_type (str): 计时器类型，可选值为 'all' 或 'epoch'。
        start_time (float): 计时器开始计时的时间戳。
    """
    # 'all' or 'epoch'
    timer_type = None
    start_time = None

    @classmethod
    def set_up(cls, timer_type: str = 'all') -> None:
        """建立计时器。

        Args:
            cls：类自己本身。
            timer_type (str): 计时器类型，可选值为 'all' 或 'epoch'，默认为'all'。
        """
        cls.timer_type = timer_type
        cls.start_time = time.time()

    @classmethod
    def clear(cls) -> None:
        """清零计时器。
        """
        cls.start_time = time.time()

    @classmethod
    def calculate_remain(cls, spend_epoch: int, spend_iteration: int, max_epoch: int, max_iteration: int, resume_epoch: int = None) -> str:
        """计算剩余时间。

        Args:
            cls：类自己本身。
            spend_epoch (int): 已经结束的epoch数。
            spend_iteration (int): 已经结束的iteration数。
            max_epoch (int): 总共的epoch数。
            max_iteration (int): 每个epoch的iteration数。
            resume_epoch (int): 恢复训练时的epoch数，可选，默认为'None'。

        Return:
            time_transform (str): 剩余时间
        """
        curr_time = time.time()
        start_time = cls.start_time
        if resume_epoch is not None:
            # 如果存在恢复训练的epoch，调整已花费和总epoch数
            spend_epoch -= resume_epoch
            max_epoch -= resume_epoch
        if cls.timer_type == 'all':
            # 如果计时器类型是'all'，计算整体剩余时间
            spend_iter_all = (spend_epoch - 1) * max_iteration + spend_iteration
            max_iter_all = max_epoch * max_iteration
            remain_time = (curr_time - start_time) / spend_iter_all * max_iter_all - (curr_time - start_time)
        elif cls.timer_type == 'epoch':
            # 如果计时器类型是'epoch'，计算当前epoch的剩余时间
            remain_time = (curr_time - start_time) / spend_iteration * max_iteration - (curr_time - start_time)
        return cls.time_transform(remain_time)

    @classmethod
    def calculate_spend(cls) -> str:
        """计算已经花费的时间。
        Args:
            cls：类自己本身。
        Returns:
            time_transform (str): 已经花费的时间
        """
        curr_time = time.time()
        start_time = cls.start_time
        return cls.time_transform(curr_time - start_time)


    @classmethod
    def time_transform(cls, time: float) -> str:
        """将时间转换为格式化字符串。

        Args:
            cls：类自己本身。
            time (float): 时间，单位为秒。

        Returns:
            time (str): 格式化的时间字符串，格式为 'Ad Bh Cm Ds'。
        """
         if time < 60:
            # 如果时间小于60秒，将时间取整作为秒数
            second = int(time)
            return str(second) + 's'
        elif time < 60*60:
            # 如果时间小于1小时，计算分钟和剩余秒数
            minute = int(time) // 60
            second = int(time) - minute * 60
            return str(minute) + 'm ' + str(second) + 's'
        elif time < 60*60*24:
            # 如果时间小于1天，计算小时、分钟和剩余秒数
            hour = int(time) // (60*60)
            minute = (int(time) - hour * (60*60)) // 60
            second = int(time) - hour * (60*60) - minute * 60
            return str(hour) + 'h ' + str(minute) + 'm ' + str(second) + 's'
        else:
            # 如果时间大于等于1天，计算天数、小时、分钟和剩余秒数
            day = int(time) // (60*60*24)
            hour = (int(time) - day * (60*60*24)) // (60*60)
            minute = (int(time) - day * (60*60*24) - hour * (60*60)) // 60
            second = int(time) - day * (60*60*24) - hour * (60*60) - minute * 60
            return str(day) + 'd ' + str(hour) + 'h ' + str(minute) + 'm ' + str(second) + 's'

            


    