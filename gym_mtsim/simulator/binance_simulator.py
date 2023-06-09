from typing import List, Tuple, Dict, Any, Optional

import os
import pickle
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from gym_mtsim.data import BINANCE_SYMBOL_PATH, BINANCE_SYMBOL_CSV_PATH

from ..metatrader import Timeframe, SymbolInfo, _local2utc
from .order import OrderType, Order
from .exceptions import SymbolNotFound, OrderNotFound
from .symbol_storage import SymbolStorage

class BinanceSimulator:

    def __init__(
            self, unit: str='USDT', balance: float=10000., leverage: float=20.,
            stop_out_level: float=0.2, hedge: bool=True, symbols_filename: Optional[str]=None, symbols_info_filename: str = BINANCE_SYMBOL_PATH
            , csvDataFolder: str = '', risk_ratio: float = 0.2
        ) -> None:
        self.symbols_info_filename = symbols_info_filename
        self.csvDataFolder = csvDataFolder if csvDataFolder else BINANCE_SYMBOL_CSV_PATH
        self.unit = unit
        self.balance = balance
        self.equity = balance
        self.margin = 0.
        self.leverage = leverage
        self.stop_out_level = stop_out_level
        self.hedge = hedge

        self.symbols_info: Dict[str, SymbolInfo] = {}
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.symbols_data_normalized: Dict[str, pd.DataFrame] = {}
        self.orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.current_time: datetime = NotImplemented
        self.risk_ratio = risk_ratio
        self.symbols_filename = symbols_filename
        try:
            if symbols_filename:
                if not self.load_symbols(symbols_filename):
                    raise FileNotFoundError(f"file '{symbols_filename}' not found")
        except Exception as e: print(e)

    @property
    def free_margin(self) -> float:
        return self.equity - self.margin


    @property
    def margin_level(self) -> float:
        margin = round(self.margin, 6)
        if margin == 0.:
            return np.inf
        return self.equity / margin

    def retrieve_data(self,
        symbol: str, from_dt: datetime, to_dt: datetime, timeframe: Timeframe
    ) -> Tuple[SymbolInfo, pd.DataFrame, pd.DataFrame]:
        with open(self.symbols_info_filename, 'r') as f:
            data = json.load(f)

        symbols = [SymbolInfo(**d) for d in data]

        symbol_info = next(filter(lambda s: s.name == symbol, symbols), None)

        utc_from = _local2utc(from_dt)
        utc_to = _local2utc(to_dt)
        
        #'./BTCUSDT/M1.csv'
        df = pd.read_csv(os.path.join(self.csvDataFolder, symbol, f'{timeframe.name}.csv'))
        #'./BTCUSDT/M1_normalized.csv'
        df_normalized = pd.read_csv(os.path.join(self.csvDataFolder, symbol, f'{timeframe.name}_normalized.csv'))
        df = df.dropna()
        df['Time'] = pd.to_datetime(df['Time'], utc=True)
        df_normalized['Time'] = pd.to_datetime(df_normalized['Time'], utc=True)

        df_normalized = df_normalized[df_normalized['Time'].between(utc_from, utc_to)]
        firstDate = df_normalized['Time'][0]
        lastDate = df_normalized['Time'].iloc[-1]
        df = df[df['Time'].between(firstDate, lastDate)]
        
        data = df[['Time', 'Open', 'Close', 'Low', 'High', 'Volume']].set_index('Time')
        data = data.loc[~data.index.duplicated(keep='first')]
        data_nomalize = df_normalized.loc[~df_normalized.index.duplicated(keep='first')].set_index('Time')

        return symbol_info, data, data_nomalize
    
    def download_data(
            self, symbols: List[str], time_range: Tuple[datetime, datetime], timeframe: Timeframe
        ) -> None:
        from_dt, to_dt = time_range
        for symbol in symbols:
            si, df, df_normalized = self.retrieve_data(symbol, from_dt, to_dt, timeframe)
            self.symbols_info[symbol] = si
            self.symbols_data[symbol] = df
            self.symbols_data_normalized[symbol] = df_normalized


    def save_symbols(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump((self.symbols_info, self.symbols_data, self.symbols_data_normalized), file)


    def load_symbols(self, filename: str) -> bool:
        if not os.path.exists(filename):
            return False
        data = SymbolStorage.get_data(filename)
        self.symbols_info = data[0]
        self.symbols_data = data[1]
        if(len(data) > 2):
            self.symbols_data_normalized = data[2]

        return True


    def tick(self, delta_time: timedelta=timedelta()) -> None:
        self._check_current_time()

        self.current_time += delta_time
        self.equity = self.balance

        for order in self.orders:
            order.exit_time = self.current_time
            order.exit_price = self.price_at(order.symbol, order.exit_time)['Close']
            self._update_order_profit(order)
            self.equity += order.profit
            

        while self.margin_level < self.stop_out_level and len(self.orders) > 0:
            most_unprofitable_order = min(self.orders, key=lambda order: order.profit)
            self.close_order(most_unprofitable_order)

        if self.balance < 0.:
            self.balance = 0.
            self.equity = self.balance


    def nearest_time(self, symbol: str, time: datetime) -> datetime:
        df = self.symbols_data[symbol]
        if time in df.index:
            return time
        try:
            i, = df.index.get_indexer([time], method='ffill')
        except KeyError:
            i, = df.index.get_indexer([time], method='bfill')
        return df.index[i]
    
    def nearest_times(self, symbol: str, time: np.ndarray) -> np.ndarray:
        df = self.symbols_data[symbol]
        result = np.zeros_like(time)
        time_df = pd.DataFrame(index=time)

        # left join with original DataFrame
        merged_df = time_df.merge(df, how='left', left_index=True, right_index=True, indicator=True)
        mask = merged_df['_merge'] == 'both'
        result[mask] = time[mask]

        try:
            indexer = df.index.get_indexer(time[~mask], method='ffill')
        except KeyError:
            indexer = df.index.get_indexer(time[~mask], method='bfill')

        result[~mask] = df.index[indexer]
        return result

    def price_at(self, symbol: str, time: datetime) -> pd.Series:
        df = self.symbols_data[symbol]
        time = self.nearest_time(symbol, time)
        return df.loc[time]
    
    def prices_at(self, symbol: str, time: List[datetime]) -> pd.Series:
        df = self.symbols_data[symbol]
        time = np.array(time)
        time = self.nearest_times(symbol, time)
        return df.loc[time]
    
    def features_at(self, symbol: str, time: datetime) -> pd.Series:
        df = self.symbols_data_normalized[symbol]
        time = np.array(time)
        time = self.nearest_times(symbol, time)
        return df.loc[time]

    def symbol_orders(self, symbol: str) -> List[Order]:
        symbol_orders = list(filter(
            lambda order: order.symbol == symbol, self.orders
        ))
        return symbol_orders


    def create_order(self, order_type: OrderType, symbol: str, volume: float, fee_rate: float=0.0005) -> Order:
        self._check_current_time()
        self._check_volume(symbol, volume)
        if fee_rate < 0.:
            raise ValueError(f"negative fee rate '{fee_rate}'")

        if self.hedge:
            return self._create_hedged_order(order_type, symbol, volume, fee_rate)
        return self._create_unhedged_order(order_type, symbol, volume, fee_rate)


    def _create_hedged_order(self, order_type: OrderType, symbol: str, volume: float, fee_rate: float) -> Order:
        order_id = len(self.closed_orders) + len(self.orders) + 1
        entry_time = self.current_time
        entry_price = self.price_at(symbol, entry_time)['Close']
        exit_time = entry_time
        exit_price = entry_price
        entry_balance = self.equity
        order_fee = fee_rate * (volume * entry_price)
        take_profit_at, stop_loss_at = self._calculate_takeprofit_and_stoploss(entry_price, volume, entry_balance)

        order = Order(
            order_id, order_type, symbol, volume, fee_rate, order_fee,
            entry_time, entry_price, exit_time, exit_price, entry_balance,
            take_profit_at, stop_loss_at
        )
        self._update_order_profit(order)
        self._update_order_margin(order)

        if order.margin > self.free_margin + order.profit:
            raise ValueError(
                f"low free margin (order margin={order.margin}, order profit={order.profit}, "
                f"free margin={self.free_margin})"
            )

        self.equity += order.profit
        self.margin += order.margin
        self.orders.append(order)
        return order


    def _create_unhedged_order(self, order_type: OrderType, symbol: str, volume: float, fee_rate: float) -> Order:
        if symbol not in map(lambda order: order.symbol, self.orders):
            return self._create_hedged_order(order_type, symbol, volume, fee_rate)

        old_order: Order = self.symbol_orders(symbol)[0]

        if old_order.type == order_type:
            new_order = self._create_hedged_order(order_type, symbol, volume, fee_rate)
            self.orders.remove(new_order)

            entry_price_weighted_average = np.average(
                [old_order.entry_price, new_order.entry_price],
                weights=[old_order.volume, new_order.volume]
            )

            old_order.volume += new_order.volume
            old_order.profit += new_order.profit
            old_order.dragdown = min([old_order.profit, old_order.dragdown])
            old_order.highestprofit = max([old_order.profit, old_order.highestprofit])
            old_order.margin += new_order.margin
            old_order.entry_price = entry_price_weighted_average
            old_order.fee_rate = max(old_order.fee_rate, new_order.fee_rate)
            old_order.fee += new_order.fee
            
            old_order.take_profit_at, old_order.stop_loss_at = self._calculate_takeprofit_and_stoploss(entry_price_weighted_average, old_order.volume, self.balance)
            return old_order

        if volume >= old_order.volume:
             self.close_order(old_order)
             if volume > old_order.volume:
                 return self._create_hedged_order(order_type, symbol, volume - old_order.volume, fee_rate)
             return old_order

        partial_profit = (volume / old_order.volume) * old_order.profit
        partial_margin = (volume / old_order.volume) * old_order.margin

        old_order.volume -= volume
        old_order.profit -= partial_profit
        old_order.margin -= partial_margin

        self.balance += partial_profit
        self.margin -= partial_margin

        return old_order


    def close_order(self, order: Order) -> float:
        self._check_current_time()
        if order not in self.orders:
            raise OrderNotFound("order not found in the order list")

        order.exit_time = self.current_time
        order.exit_price = self.price_at(order.symbol, order.exit_time)['Close']
        order.fee += order.fee_rate * order.volume * order.exit_price

        self._update_order_profit(order)

        self.balance += order.profit
        self.margin -= order.margin

        order.closed = True
        self.orders.remove(order)
        self.closed_orders.append(order)
        return order.profit


    def get_state(self) -> Dict[str, Any]:
        orders = []
        for order in reversed(self.closed_orders + self.orders):
            orders.append({
                'Id': order.id,
                'Symbol': order.symbol,
                'Type': order.type.name,
                'Volume': order.volume,
                'Entry Time': order.entry_time,
                'Entry Price': order.entry_price,
                'Exit Time': order.exit_time,
                'Exit Price': order.exit_price,
                'Profit': order.profit,
                "Dragdown": order.dragdown,
                "HighestProfit": order.highestprofit,
                'Margin': order.margin,
                'Fee': order.fee,
                'Closed': order.closed,
            })
        orders_df = pd.DataFrame(orders)

        return {
            'current_time': self.current_time,
            'balance': self.balance,
            'equity': self.equity,
            'margin': self.margin,
            'free_margin': self.free_margin,
            'margin_level': self.margin_level,
            'orders': orders_df,
        }


    def _update_order_profit(self, order: Order) -> None:
        diff = order.exit_price - order.entry_price
        v = order.volume * self.symbols_info[order.symbol].trade_contract_size
        local_profit = v * (order.type.sign * diff)

        order.profit = local_profit * self._get_unit_ratio(order.symbol, order.exit_time)
        order.profit -= order.fee

        order.dragdown = min([order.dragdown, order.profit])
        order.highestprofit = max([order.highestprofit, order.profit])


    def _update_order_margin(self, order: Order) -> None:
        v = order.volume * self.symbols_info[order.symbol].trade_contract_size

        local_margin = (v * order.entry_price) / self.leverage
        local_margin *= self.symbols_info[order.symbol].margin_rate
        
        order.margin = local_margin * self._get_unit_ratio(order.symbol, order.entry_time)


    def _get_unit_ratio(self, symbol: str, time: datetime) -> float:
        symbol_info = self.symbols_info[symbol]
        if self.unit == symbol_info.currency_profit:
            return 1.

        if self.unit == symbol_info.currency_margin:
            return 1 / self.price_at(symbol, time)['Close']

        currency = symbol_info.currency_profit
        unit_symbol_info = self._get_unit_symbol_info(currency)
        if unit_symbol_info is None:
            raise SymbolNotFound(f"unit symbol for '{currency}' not found")

        unit_price = self.price_at(unit_symbol_info.name, time)['Close']
        if unit_symbol_info.currency_margin == self.unit:
            unit_price = 1. / unit_price

        return unit_price


    def _get_unit_symbol_info(self, currency: str) -> Optional[SymbolInfo]:  # Unit/Currency or Currency/Unit
        for info in self.symbols_info.values():
            if currency in info.currencies and self.unit in info.currencies:
                return info
        return None


    def _check_current_time(self) -> None:
        if self.current_time is NotImplemented:
            raise ValueError("'current_time' must have a value")


    def _check_volume(self, symbol: str, volume: float) -> None:
        symbol_info = self.symbols_info[symbol]
        if not (symbol_info.volume_min <= volume <= symbol_info.volume_max):
            raise ValueError(
                f"'volume' must be in range [{symbol_info.volume_min}, {symbol_info.volume_max}]"
            )
        if not round(volume / symbol_info.volume_step, 6).is_integer():
            raise ValueError(f"'volume' must be a multiple of {symbol_info.volume_step}")
        
    def _calculate_takeprofit_and_stoploss(self, entry_price : float, volume : float, balance : float):
        max_risk_per_trade = self.risk_ratio * -balance

        take_profit_at = (volume / self.leverage) * entry_price
        stop_loss_at = take_profit_at / -2
        if(stop_loss_at < max_risk_per_trade):
            stop_loss_at = max_risk_per_trade
            take_profit_at = stop_loss_at * -2
        
        return take_profit_at, stop_loss_at