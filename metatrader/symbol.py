from typing import Tuple
from dataclasses import dataclass
import json

from .interface import MtSymbolInfo

@dataclass
class SymbolInfo:
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
    
    def __init__(self, info: MtSymbolInfo) -> None:
        self.name: str = info.name
        self.market: str = self._get_market(info)

        self.currency_margin: str = info.currency_margin
        self.currency_profit: str = info.currency_profit
        self.currencies: Tuple[str, ...] = tuple(set([self.currency_margin, self.currency_profit]))

        self.trade_contract_size: float = info.trade_contract_size
        self.margin_rate: float = 1.0  # MetaTrader info does not contain this value!

        self.volume_min: float = info.volume_min
        self.volume_max: float = info.volume_max
        self.volume_step: float = info.volume_step

    def __init__(self, name, market, currency_margin, currency_profit, currencies, trade_contract_size,
                 margin_rate, volume_min, volume_max, volume_step):
        self.name = name
        self.market = market
        self.currency_margin = currency_margin
        self.currency_profit = currency_profit
        self.currencies = tuple(currencies) if currencies is not None else tuple(set([currency_margin, currency_profit]))
        self.trade_contract_size = trade_contract_size
        self.margin_rate = margin_rate
        self.volume_min = volume_min
        self.volume_max = volume_max
        self.volume_step = volume_step
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)
    
    def __str__(self) -> str:
        return f'{self.market}/{self.name}'


    def _get_market(self, info: MtSymbolInfo) -> str:
        mapping = {
            'forex': 'Forex',
            'crypto': 'Crypto',
            'stock': 'Stock',
        }

        root = info.path.split('\\')[0]
        for k, v in mapping.items():
            if root.lower().startswith(k):
                return v

        return root
