from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import copy
from datetime import datetime, timedelta
from plotly.graph_objects import Figure

import numpy as np
from scipy.special import expit

import random
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_colors
import plotly.graph_objects as go
import os
import sys
if 'gymnasium' in sys.modules:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding
else:
    import gym
    from gym import spaces
    from gym.utils import seeding


from ..simulator import BinanceSimulator, OrderType

class MtEnv(gym.Env):

    metadata = {'render.modes': ['human', 'simple_figure', 'advanced_figure']}

    def __init__(
            self, original_simulator: BinanceSimulator, trading_symbols: List[str],
            window_size: int, time_points: Optional[List[datetime]]=None,
            hold_threshold: float=0.5, close_threshold: float=0.5,
            fee: Union[float, Callable[[str], float]]=0.0005, env_size: int = 200,
            symbol_max_orders: int=1, save_img_in_reset: bool=False, old_gym: bool = False
        ) -> None:

        # validations
        assert len(original_simulator.symbols_data) > 0, "no data available"
        assert len(original_simulator.symbols_info) > 0, "no data available"
        assert len(trading_symbols) > 0, "no trading symbols provided"
        assert 0. <= hold_threshold <= 1., "'hold_threshold' must be in range [0., 1.]"

        if not original_simulator.hedge:
            symbol_max_orders = 1

        for symbol in trading_symbols:
            assert symbol in original_simulator.symbols_info, f"symbol '{symbol}' not found"
            currency_profit = original_simulator.symbols_info[symbol].currency_profit
            assert original_simulator._get_unit_symbol_info(currency_profit) is not None, \
                   f"unit symbol for '{currency_profit}' not found"

        if time_points is None:
            try:
                import cudf
                datetime_index = original_simulator.symbols_data[trading_symbols[0]].index
                time_points = datetime_index.astype('datetime64[ms]').astype('int64').to_arrow().tolist()
            except ImportError:
                time_points = original_simulator.symbols_data[trading_symbols[0]].index.to_pydatetime().tolist()
        
        assert len(time_points) > window_size, "not enough time points provided"

        # attributes
        self.seed()
        self.original_simulator = original_simulator
        self.trading_symbols = trading_symbols
        self.env_size = env_size
        self.window_size = window_size
        self.time_points = time_points
        self.hold_threshold = hold_threshold
        self.close_threshold = close_threshold
        self.fee = fee
        self.symbol_max_orders = symbol_max_orders
        self._is_dead = 0
        self.prices = self._get_prices()
        self.signal_features = self._process_data()
        self.features_shape = (window_size, self.signal_features.shape[1])
        self.initBalance = original_simulator.balance
        self.save_img_in_reset = save_img_in_reset
        self.old_gym = old_gym
        # spaces
        self.action_space = spaces.Box(
            low=-1e9, high=1e9, dtype=np.float32,
            shape=(len(self.trading_symbols) * (self.symbol_max_orders + 2),)
        )  # symbol -> [close_order_i(logit), hold(logit), volume]

        # self.observation_space = spaces.Dict({
        #     'balance': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'equity': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'margin': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'is_dead': spaces.Discrete(2), # 0 = not dead, 1 = dead
        #     'features': spaces.Box(low=-np.inf, high=np.inf, shape=self.features_shape, dtype=np.float32),
        #     'orders': spaces.Box(
        #         low=-np.inf, high=np.inf, dtype=np.float32,
        #         shape=(len(self.trading_symbols), self.symbol_max_orders, 3)
        #     ),  # symbol, order_i -> [entry_price, volume, profit]
        # })
        # Flatten the 'orders' component
        flattened_orders_shape = (len(self.trading_symbols) * self.symbol_max_orders * 3,)

        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'equity': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'margin': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'is_dead': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=self.features_shape, dtype=np.float32),
            'orders': spaces.Box(
                low=-np.inf, high=np.inf, dtype=np.float32,
                shape=flattened_orders_shape
            ),  # symbol, order_i -> [entry_price, volume, profit]
        })
        # episode
        self._start_tick, self._end_tick = self.getEnvironmentRange()

        self._done: bool = NotImplemented
        self._current_tick: int = NotImplemented
        self.simulator: BinanceSimulator = NotImplemented
        self.history: List[Dict[str, Any]] = NotImplemented


    def seed(self, seed: Optional[int]=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getEnvironmentRange(self):
        start_tick = self.window_size - 1
        end_tick = len(self.time_points) - 1

        start_tick = random.randint(start_tick, end_tick - self.env_size)
        end_tick = start_tick + self.env_size
        
        return start_tick, end_tick

    def reset(self, *, seed=None, options=None) -> Dict[str, np.ndarray]:
        if not self.old_gym:
            super().reset(seed=seed)

        try:
            if(self.save_img_in_reset and os.path.exists("img_log") and self.history != NotImplemented and len(self.history) > 0):
                fig = self.render('advanced_figure', time_format="%Y-%m-%d", return_figure = True)
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fig.write_image(f"img_log/log_{self.simulator.balance}_{timestamp_str}.png")
                fig.write_html(f"img_log/log_{self.simulator.balance}_{timestamp_str}.html")
        except:
            print('Exception at reset write log')
        self._done = False
        self._is_dead = 0
        self._start_tick, self._end_tick = self.getEnvironmentRange()
        self._current_tick = self._start_tick
        self.simulator = copy.deepcopy(self.original_simulator)
        self.simulator.current_time = self.time_points[self._current_tick]
        self.history = [self._create_info(step_reward = 0, reward_description = "")]
        return self._get_observation(), self.history[0]


    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        action = self._normalize_action(action)
        #print(action)
        orders_info, closed_orders_info = self._apply_action(action)

        self._current_tick += 1
        if self._current_tick == self._end_tick:                
            self._done = True

        dt = self.time_points[self._current_tick] - self.time_points[self._current_tick - 1]
        self.simulator.tick(dt)

        step_reward, reward_description = self._calculate_reward()

        info = self._create_info(
            orders=orders_info, closed_orders=closed_orders_info, step_reward=step_reward,
            reward_description = reward_description
        )
        observation = self._get_observation()
        self.history.append(info)
        if(self._is_dead):
            self._done = True
        truncated = False

        if not self.old_gym:
            return observation, step_reward, self._done, truncated, info
        else:
            return observation, step_reward, self._done, info 


    def _apply_action(self, action: np.ndarray) -> Tuple[Dict, Dict]:
        orders_info = {}
        closed_orders_info = {symbol: [] for symbol in self.trading_symbols}

        k = self.symbol_max_orders + 2

        for i, symbol in enumerate(self.trading_symbols):
            symbol_action = action[k*i:k*(i+1)]
            close_orders_logit = symbol_action[:-2]
            hold_logit = symbol_action[-2]
            volume = symbol_action[-1]
            currentClose = self.simulator.price_at(symbol, self.time_points[self._current_tick])['Close']
            volume = volume * self.simulator.equity / currentClose
            
            close_orders_probability = abs(close_orders_logit)
            hold_probability = abs(hold_logit)
            hold = bool(hold_probability > self.hold_threshold)
            modified_volume = self._get_modified_volume(symbol, volume)

            symbol_orders = self.simulator.symbol_orders(symbol)
            orders_to_close_index = np.where(
                close_orders_probability[:len(symbol_orders)] > self.close_threshold
            )[0]
            orders_to_close = np.array(symbol_orders)[orders_to_close_index]

            for j, order in enumerate(orders_to_close):
                self.simulator.close_order(order)
                closed_orders_info[symbol].append(dict(
                    order_id=order.id, symbol=order.symbol, order_type=order.type,
                    volume=order.volume, fee=order.fee,
                    margin=order.margin, profit=order.profit,
                    close_probability=close_orders_probability[orders_to_close_index][j],
                ))

            orders_capacity = self.symbol_max_orders - (len(symbol_orders) - len(orders_to_close))
            orders_info[symbol] = dict(
                order_id=None, symbol=symbol, hold_probability=hold_probability,
                hold=hold, volume=volume, capacity=orders_capacity, order_type=None,
                modified_volume=modified_volume, fee=float('nan'), margin=float('nan'),
                error='',
            )

            if self.simulator.hedge and orders_capacity == 0:
                orders_info[symbol].update(dict(
                    error="cannot add more orders"
                ))
            elif not hold:
                order_type = OrderType.Buy if volume > 0. else OrderType.Sell
                fee = self.fee if type(self.fee) is float else self.fee(symbol)

                try:
                    order = self.simulator.create_order(order_type, symbol, modified_volume, fee)
                    new_info = dict(
                        order_id=order.id, order_type=order_type,
                        fee=fee, margin=order.margin,
                    )
                except ValueError as e:
                    new_info = dict(error=str(e))

                orders_info[symbol].update(new_info)

        return orders_info, closed_orders_info


    def _get_prices(self, keys: List[str]=['Close', 'Open']) -> Dict[str, np.ndarray]:
        prices = {}

        for symbol in self.trading_symbols:
            p = self.original_simulator.prices_at(symbol, self.time_points)[keys]
            prices[symbol] = np.array(p)

        return prices

    def _get_features(self) -> Dict[str, np.ndarray]:
        data = None
        if not hasattr(self.original_simulator, 'symbols_data_normalized'):
            data = self.prices
        else:
            features = {}
            for symbol in self.trading_symbols:
                keys = self.original_simulator.symbols_data_normalized[symbol].columns.tolist()
                remove = ['Time', 'normalizesma']
                keys = list(filter(lambda x: x not in remove, keys))
                p = self.original_simulator.features_at(symbol, self.time_points)[keys]
                features[symbol] = np.array(p)

            data = features
        return data
    
    def _process_data(self) -> np.ndarray:
        data = self._get_features()
        signal_features = np.column_stack(list(data.values()))
        return signal_features


    def _get_observation(self) -> Dict[str, np.ndarray]:
        features = self.signal_features[(self._current_tick-self.window_size+1):(self._current_tick+1)]

        # orders = np.zeros((len(self.trading_symbols), self.symbol_max_orders, 3))
        # for i, symbol in enumerate(self.trading_symbols):
        #     symbol_orders = self.simulator.symbol_orders(symbol)
        #     for j, order in enumerate(symbol_orders):
        #         orders[i, j] = [order.entry_price, order.volume, order.profit]

        orders = np.zeros(np.prod(self.observation_space['orders'].shape))
        orders_shape = [len(self.trading_symbols), self.symbol_max_orders, 3]
 
        for i, symbol in enumerate(self.trading_symbols):
            symbol_orders = self.simulator.symbol_orders(symbol)
            for j, order in enumerate(symbol_orders):
                index = i * orders_shape[0] * (orders_shape[2] + orders_shape[1]) + j * orders_shape[2]
                orders[index:(index+3)] = [order.entry_price, order.volume, order.profit]

        # 2 symbol, 4 max, 3 shape
        #0 0 0, 0 0 0, 0 0 0, 0 0 0, 1 1 1, 1 1 1, 1 1 1, 1 1 

        observation = {
            'balance': np.array([self.simulator.balance]),
            'equity': np.array([self.simulator.equity]),
            'margin': np.array([self.simulator.margin]),
            'is_dead': np.array([self._is_dead]),
            'features': features,
            'orders': orders,
        }
        return observation


    def _calculate_reward(self) -> float:
        prev_equity = self.history[-1]['equity']
        current_equity = self.simulator.equity
        step_reward = 0
        reward_description = ''

        bonus_open_order_deduct_ratio = -0.005
        take_profit_reached_ratio = 0.3
        stop_loss_reached_ratio = 0.3
        holding_order_weight = 0.1
        equity_change_weight = 3
        rate_of_return_weight = 2
        penalty_for_overleveraging_weight = 0.1

        # Check if the agent has run out of money
        if current_equity <= 0:
            step_reward = -100
            self._is_dead = 1
        else:
            # Bonus for opening a new order
            bonus_for_open_order = 0.
            if len(self.simulator.orders) == 0:
                bonus_for_open_order = random.uniform(bonus_open_order_deduct_ratio, 0)

            step_reward += bonus_for_open_order
            reward_description += f"Bonus (+): {bonus_for_open_order}<br> "
            # Penalty for holding orders for too long
            penalty_for_holding_order = 0
            for order in self.simulator.orders:
                penalty_for_holding_order += self._bar_between_interval(order.entry_time, self.simulator.current_time)
            
            penalty_for_holding_order *= holding_order_weight
            step_reward -= penalty_for_holding_order
            reward_description += f"Penalty hold (-): {penalty_for_holding_order}<br> "

            # 
            for order in self.simulator.orders:
                if(order.profit >= order.take_profit_at):
                    tp_reach_bonus = self._calculate_ratio(order.take_profit_at, order.profit)
                    tp_reach_bonus *= take_profit_reached_ratio
                    step_reward += tp_reach_bonus
                    reward_description += f"Tp reached bonus (+): {tp_reach_bonus}<br> "
                elif(order.profit <= order.stop_loss_at):
                    sl_reach_bonus = self._calculate_ratio(abs(order.stop_loss_at), abs(order.profit))
                    sl_reach_bonus *= stop_loss_reached_ratio
                    step_reward += sl_reach_bonus
                    reward_description += f"SL reached bonus (+): {sl_reach_bonus}<br> "

            total_profit = 0.0
            # Calculate the profit from closed orders
            for order in self.simulator.closed_orders:
                if order.exit_time == self.simulator.current_time:
                    # Calculate the profit as a ratio of the entry balance
                    profit = self._calculate_ratio(order.entry_balance + order.profit, order.entry_balance)
                    is_loss = 1 if profit < 0 else 0
                    profit *= (1 - 0.2 * is_loss) 

                    # Calculate the dragdown as a ratio of the entry balance
                    dragdown = self._calculate_ratio(order.entry_balance + order.dragdown, order.entry_balance)
                    
                    # Add a penalty if the agent closed the order at the maximum dragdown
                    if order.profit < 0 and order.profit == order.dragdown:
                        dragdown += dragdown * 0.1

                    reward_description += f"Closed profit (+): {profit}<br> "
                    reward_description += f"Closed dragdown (+): {dragdown}<br> "
                    total_profit += profit + dragdown
                    
                    if(order.highestprofit > 0):
                        profit = order.profit / order.highestprofit
                        if profit > 0.5:
                            total_profit += 0.1 * profit
                            reward_description += f"Closed profit (highest) (+): {0.1 * profit}<br> "

                        dragdown = order.dragdown / order.highestprofit
                        if dragdown < -0.5:
                            total_profit -= 0.1 * abs(dragdown)
                            reward_description += f"Closed dragdown (highest) (-): { 0.1 * abs(dragdown)}<br> "

            # Add the profit and equity change to the step reward
            step_reward += total_profit
            equity_change = self._calculate_ratio(current_equity, prev_equity)
            # Focus on equity change
            equity_change *= equity_change_weight
            step_reward += equity_change
            reward_description += f"Equity change (+): {equity_change}<br> "

            balance_changes = self._calculate_ratio(self.simulator.balance, self.original_simulator.balance)
            step_reward += balance_changes
            reward_description += f"Balance change (+): {balance_changes}<br> "

            # Add a reward for maintaining a high rate of return
            rate_of_return = self._calculate_ratio(current_equity, self.simulator.balance)
            rate_of_return *= rate_of_return_weight
            step_reward += rate_of_return
            reward_description += f"Rate of return (+): {rate_of_return}<br> "

            # # Add a penalty for overleveraging
            if self.simulator.margin / self.simulator.equity > 0.9:
                penalty_for_overleveraging = self.simulator.margin / self.simulator.equity
                penalty_for_overleveraging *= penalty_for_overleveraging_weight
                step_reward -= penalty_for_overleveraging
                reward_description += f"Overleveraging penalty (-): {penalty_for_overleveraging}<br> "

        return step_reward, reward_description


    def _create_info(self, **kwargs: Any) -> Dict[str, Any]:
        info = {k: v for k, v in kwargs.items()}
        info['balance'] = self.simulator.balance
        info['equity'] = self.simulator.equity
        info['margin'] = self.simulator.margin
        info['free_margin'] = self.simulator.free_margin
        info['margin_level'] = self.simulator.margin_level
        return info


    def _get_modified_volume(self, symbol: str, volume: float) -> float:
        si = self.simulator.symbols_info[symbol]
        v = abs(volume)
        v = np.clip(v, si.volume_min, si.volume_max)
        v = round(v / si.volume_step) * si.volume_step
        return v


    def render(self, mode: str='human', **kwargs: Any) -> Figure:
        if mode == 'simple_figure':
            return self._render_simple_figure(**kwargs)
        if mode == 'advanced_figure':
            return self._render_advanced_figure(**kwargs)
        return self.simulator.get_state(**kwargs)


    def _render_simple_figure(
        self, figsize: Tuple[float, float]=(14, 6), return_figure: bool=False
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            draw_price = self.prices[symbol][:, 0][self._start_tick:self._end_tick]
            draw_points = self.time_points[self._start_tick:self._end_tick]
            symbol_color = symbol_colors[j]

            ax.plot(draw_points, draw_price, c=symbol_color, marker='.', label=symbol)

            buy_ticks = []
            buy_error_ticks = []
            sell_ticks = []
            sell_error_ticks = []
            close_ticks = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol, {})
                if order and not order['hold']:
                    if order['order_type'] == OrderType.Buy:
                        if order['error']:
                            buy_error_ticks.append(tick)
                        else:
                            buy_ticks.append(tick)
                    else:
                        if order['error']:
                            sell_error_ticks.append(tick)
                        else:
                            sell_ticks.append(tick)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    close_ticks.append(tick)

            tp = np.array(self.time_points)
            ax.plot(tp[buy_ticks], close_price[buy_ticks], '^', color='green')
            ax.plot(tp[buy_error_ticks], close_price[buy_error_ticks], '^', color='gray')
            ax.plot(tp[sell_ticks], close_price[sell_ticks], 'v', color='red')
            ax.plot(tp[sell_error_ticks], close_price[sell_error_ticks], 'v', color='gray')
            ax.plot(tp[close_ticks], close_price[close_ticks], '|', color='black')

            ax.tick_params(axis='y', labelcolor=symbol_color)
            ax.yaxis.tick_left()
            if j < len(self.trading_symbols) - 1:
                ax = ax.twinx()

        fig.suptitle(
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.legend(loc='right')

        if return_figure:
            return fig

        plt.show()


    def _render_advanced_figure(
            self, figsize: Tuple[float, float]=(1400, 600), time_format: str="%Y-%m-%d %H:%m",
            return_figure: bool=False
        ) -> Any:

        fig = go.Figure()

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))
        get_color_string = lambda color: "rgba(%s, %s, %s, %s)" % tuple(color)

        extra_info = [
            f"balance: {h['balance']:.6f} {self.simulator.unit}<br>"
            f"equity: {h['equity']:.6f}<br>"
            f"margin: {h['margin']:.6f}<br>"
            f"free margin: {h['free_margin']:.6f}<br>"
            f"margin level: {h['margin_level']:.6f}<br>"
            f"reward: {h['step_reward']:.6f}<br>"
            f"reward_desc: {h['reward_description']}"
            for h in self.history
        ]

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            draw_price = self.prices[symbol][:, 0][self._start_tick:self._end_tick]
            draw_points = self.time_points[self._start_tick:self._end_tick]
            symbol_color = symbol_colors[j]

            fig.add_trace(
                go.Scatter(
                    x=draw_points,
                    y=draw_price,
                    mode='lines+markers',
                    line_color=get_color_string(symbol_color),
                    opacity=1.0,
                    hovertext=extra_info,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.update_layout(**{
                f'yaxis{j+1}': dict(
                    tickfont=dict(color=get_color_string(symbol_color * [1, 1, 1, 0.8])),
                    overlaying='y' if j > 0 else None,
                    # position=0.035*j
                ),
            })

            trade_ticks = []
            trade_markers = []
            trade_colors = []
            trade_sizes = []
            trade_extra_info = []
            trade_max_volume = max([
                h.get('orders', {}).get(symbol, {}).get('modified_volume') or 0
                for h in self.history
            ])
            close_ticks = []
            close_extra_info = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol)
                if order and not order['hold']:
                    marker = None
                    color = None
                    size = 8 + 22 * (order['modified_volume'] / trade_max_volume)
                    info = (
                        f"order id: {order['order_id'] or ''}<br>"
                        f"hold probability: {order['hold_probability']:.4f}<br>"
                        f"hold: {order['hold']}<br>"
                        f"volume: {order['volume']:.6f}<br>"
                        f"modified volume: {order['modified_volume']:.4f}<br>"
                        f"fee: {order['fee']:.6f}<br>"
                        f"margin: {order['margin']:.6f}<br>"
                        f"error: {order['error']}"
                    )

                    if order['order_type'] == OrderType.Buy:
                        marker = 'triangle-up'
                        color = 'gray' if order['error'] else 'green'
                    else:
                        marker = 'triangle-down'
                        color = 'gray' if order['error'] else 'red'

                    trade_ticks.append(tick)
                    trade_markers.append(marker)
                    trade_colors.append(color)
                    trade_sizes.append(size)
                    trade_extra_info.append(info)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    info = []
                    for order in closed_orders:
                        info_i = (
                            f"order id: {order['order_id']}<br>"
                            f"order type: {order['order_type'].name}<br>"
                            f"close probability: {order['close_probability']:.4f}<br>"
                            f"margin: {order['margin']:.6f}<br>"
                            f"profit: {order['profit']:.6f}"
                        )
                        info.append(info_i)
                    info = '<br>---------------------------------<br>'.join(info)

                    close_ticks.append(tick)
                    close_extra_info.append(info)

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[trade_ticks],
                    y=close_price[trade_ticks],
                    mode='markers',
                    hovertext=trade_extra_info,
                    marker_symbol=trade_markers,
                    marker_color=trade_colors,
                    marker_size=trade_sizes,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[close_ticks],
                    y=close_price[close_ticks],
                    mode='markers',
                    hovertext=close_extra_info,
                    marker_symbol='diamond-wide',
                    marker_color='yellow',
                    marker_size=7,
                    marker_line_width=2,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

        title = (
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f} ~ "
        )
        for j, symbol in enumerate(self.trading_symbols):
            total_reward = sum(
                [ h['step_reward']  for h in self.history ]
                )
            title += f"<br>Reward {symbol}: {total_reward:.6f}"
            
        fig.update_layout(
            title=title,
            xaxis_tickformat=time_format,
            width=figsize[0],
            height=figsize[1],
        )

        if return_figure:
            return fig

        fig.show()


    def close(self) -> None:
        plt.close()

    def _bar_between_interval(self, start_datetime : datetime, end_datetime : datetime, interval = timedelta(days=1)):
        time_diff = end_datetime - start_datetime
        interval_diff = time_diff // interval
        return interval_diff
    
    def _calculate_ratio(self, current_value, prev_value):
        ratio = (current_value - prev_value) / prev_value
        return ratio
    
    def _normalize_action(self, action):
        formatted_str = np.array2string(action, formatter={'float_kind':
        lambda x: ('-' if x < 0 else '') + '0.' + (f'{x:.4f}' % x).replace('.', '').replace('-', '')})
        formatted_str = formatted_str.replace('[', '').replace(']', '').replace('\n', '')
        formatted_array = np.fromiter(formatted_str.split(), dtype=np.float32)
        return formatted_array