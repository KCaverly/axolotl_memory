from enum import Enum


class MemoryCategory(Enum):
    MODELLING = 1
    TRAINING = 2
    INFERENCE = 3

    def __str__(self):
        if self == MemoryCategory.MODELLING:
            return "Modelling"
        elif self == MemoryCategory.TRAINING:
            return "Training"
        elif self == MemoryCategory.INFERENCE:
            return "Inference"


class MemoryPrecision(Enum):
    BIT32 = 1
    BIT16 = 2
    BIT8 = 4
    BIT4 = 5
    MIXED = 6
    UNKNOWN = 7

    def __str__(self):
        if self == MemoryPrecision.BIT32:
            return "BIT32"
        elif self == MemoryPrecision.BIT16:
            return "BIT16"
        elif self == MemoryPrecision.BIT8:
            return "BIT8 "
        elif self == MemoryPrecision.BIT4:
            return "BIT4 "
        elif self == MemoryPrecision.MIXED:
            return "MIXED"
        else:
            return "UNKNOWN"


class MemoryItem:
    def __init__(
        self,
        category: MemoryCategory,
        title: str,
        precision: MemoryPrecision,
        memory: float,
    ):
        self.category = category
        self.title = title
        self.precision = precision
        self.memory = memory

    def __str__(self) -> str:
        return f"{self.title}"
