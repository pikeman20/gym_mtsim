from enum import Enum
from datetime import datetime

import numpy as np

from typing import Tuple

class MtSymbolInfo:
    name: str
    market: str
    currency_margin: str
    currency_profit: str
    currencies: Tuple[str, ...]
    trade_contract_size: float
    margin_rate: float
    volume_min: float
    volume_max: float
    volume_step: float

class Timeframe(Enum):
    M1                        = 1
    M2                        = 2
    M3                        = 3
    M4                        = 4
    M5                        = 5
    M6                        = 6
    M10                       = 10
    M12                       = 12
    M15                       = 15
    M20                       = 20
    M30                       = 30
    H1                        = 1  | 0x4000
    H2                        = 2  | 0x4000
    H4                        = 4  | 0x4000
    H3                        = 3  | 0x4000
    H6                        = 6  | 0x4000
    H8                        = 8  | 0x4000
    H12                       = 12 | 0x4000
    D1                        = 24 | 0x4000
    W1                        = 1  | 0x8000
    MN1                       = 1  | 0xC000


def initialize() -> bool:
    return False


def shutdown() -> None:
    return None


def copy_rates_range(symbol: str, timeframe: Timeframe, date_from: datetime, date_to: datetime) -> np.ndarray:
    return None


def symbol_info(symbol: str) -> MtSymbolInfo:
    return None
