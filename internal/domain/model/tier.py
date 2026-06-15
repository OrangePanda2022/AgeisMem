from enum import Enum

class Tier(str, Enum):
    """存储分层等级枚举。

    根据访问热度和重要性将记忆数据分为四个层级：
    L0 - 活跃层（Active）：高频访问的热数据
    L1 - 温层（Warm）：中等频率访问的数据
    L2 - 冷层（Cold）：低频访问的数据
    L3 - 归档层（Archive）：极少访问的归档数据
    """
    L0 = "L0"  # 活跃层
    L1 = "L1"  # 温层
    L2 = "L2"  # 冷层
    L3 = "L3"  # 归档层