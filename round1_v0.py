from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK="SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK:{
        # copy from KELP
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50,Product.SQUID_INK: 50}

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume


    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    # def KELP_fair_value(self, order_depth: OrderDepth, method: str = "weighted_mid", min_vol: int = 15) -> float:
    #     """
    #     计算KELP的公平价格，支持多种计算方法
    #     method可选:
    #     - "weighted_mid": 加权中间价（抗异常订单）
    #     - "micro_price": 加入盘口深度信息的微观价格
    #     - "pure_weighted": 纯加权价格
    #     """
    #     if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
    #         return 0  # 无盘口数据时返回0
    #
    #     best_ask = min(order_depth.sell_orders.keys())
    #     best_bid = max(order_depth.buy_orders.keys())
    #
    #     if method == "weighted_mid":
    #         # 加权中间价（您的核心思路）
    #         weighted_ask = sum(p * abs(v) for p, v in order_depth.sell_orders.items()) / sum(
    #             abs(v) for v in order_depth.sell_orders.values())
    #         weighted_bid = sum(p * abs(v) for p, v in order_depth.buy_orders.items()) / sum(
    #             abs(v) for v in order_depth.buy_orders.values())
    #         return (weighted_bid + weighted_ask) / 2
    #
    #     elif method == "micro_price":
    #         # 微观价格（加入盘口压力）
    #         total_bid_vol = sum(order_depth.buy_orders.values())
    #         total_ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
    #         imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
    #         return (best_bid + best_ask) / 2 + imbalance * (best_ask - best_bid) / 2
    #
    #     elif method == "pure_weighted":
    #         # 纯加权价格（所有档位）
    #         total_notional = sum(p * abs(v) for p, v in {**order_depth.buy_orders, **order_depth.sell_orders}.items())
    #         total_volume = sum(abs(v) for v in {**order_depth.buy_orders, **order_depth.sell_orders}.values())
    #         return total_notional / total_volume
    #
    #     else:  # 默认使用带成交量过滤的中间价
    #         filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= min_vol]
    #         filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= min_vol]
    #         valid_ask = min(filtered_ask) if filtered_ask else best_ask
    #         valid_bid = max(filtered_bid) if filtered_bid else best_bid
    #         return (valid_ask + valid_bid) / 2

    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None
    #
    # def RAINFOREST_RESIN_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
    #     orders: List[Order] = []
    #
    #     buy_order_volume = 0
    #     sell_order_volume = 0
    #     # mm_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 20])
    #     # mm_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20])
    #
    #     baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
    #     bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
    #
    #     # Take Orders
    #     buy_order_volume, sell_order_volume = self.take_best_orders(Product.RAINFOREST_RESIN, fair_value, 0.5, orders, order_depth, position, buy_order_volume, sell_order_volume)
    #     # Clear Position Orders
    #     buy_order_volume, sell_order_volume = self.clear_position_order(Product.RAINFOREST_RESIN, fair_value, 1, orders, order_depth, position, buy_order_volume, sell_order_volume)
    #     # Market Make
    #     buy_order_volume, sell_order_volume = self.market_make(Product.RAINFOREST_RESIN, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
    #
    #     return orders

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    #
    # def KELP_orders(self, order_depth: OrderDepth, timespan: int, width: float,
    #                 KELP_take_width: float, position: int, position_limit: int) -> List[Order]:
    #     orders: List[Order] = []
    #     buy_order_volume = sell_order_volume = 0
    #
    #     if not order_depth.sell_orders or not order_depth.buy_orders:
    #         return orders
    #
    #     # 1. 计算加权中间价（抗异常订单）
    #     fair_value = self.KELP_fair_value(order_depth, method="weighted_mid")
    #
    #     # 2. 检测异常订单信号
    #     best_ask = min(order_depth.sell_orders.keys())
    #     best_bid = max(order_depth.buy_orders.keys())
    #     spread = best_ask - best_bid
    #
    #     # 异常订单检测条件（可调整参数）
    #     SMALL_ORDER_SIZE = 5# 小单阈值
    #     SPREAD_CHANGE_RATIO = 0.5  # 价差变化比例阈值
    #
    #     # 检查是否存在异常小单导致价差缩小
    #     if spread < (self.KELP_prices[-1] if self.KELP_prices else 3) * SPREAD_CHANGE_RATIO:
    #         for price, volume in order_depth.sell_orders.items():
    #             # 卖单异常条件：价格低于加权中间价且单量小
    #             if price < fair_value and abs(volume) <= SMALL_ORDER_SIZE:
    #                 quantity = min(abs(volume), position_limit - position - buy_order_volume)
    #                 if quantity > 0:
    #                     orders.append(Order(Product.KELP, price, quantity))
    #                     buy_order_volume += quantity
    #                     break  # 只处理最优先的异常单
    #
    #         for price, volume in order_depth.buy_orders.items():
    #             # 买单异常条件：价格高于加权中间价且单量小
    #             if price > fair_value and abs(volume) <= SMALL_ORDER_SIZE:
    #                 quantity = min(abs(volume), position_limit + position - sell_order_volume)
    #                 if quantity > 0:
    #                     orders.append(Order(Product.KELP, price, -quantity))
    #                     sell_order_volume += quantity
    #                     break
    #
    #     # 3. 常规做市逻辑（保留原有框架）
    #     aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
    #     bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
    #     baaf = min(aaf) if aaf else round(fair_value + width)
    #     bbbf = max(bbf) if bbf else round(fair_value - width)
    #
    #     # 做市报价
    #     buy_quantity = position_limit - (position + buy_order_volume)
    #     if buy_quantity > 0:
    #         orders.append(Order(Product.KELP, bbbf + 1, buy_quantity))
    #
    #     sell_quantity = position_limit + (position - sell_order_volume)
    #     if sell_quantity > 0:
    #         orders.append(Order(Product.KELP, baaf - 1, -sell_quantity))
    #
    #     # 4. 更新价格记忆（用于下次价差比较）
    #     current_mid = (best_ask + best_bid) / 2
    #     self.KELP_prices.append(current_mid)
    #     if len(self.KELP_prices) > timespan:
    #         self.KELP_prices.pop(0)
    #
    #     return orders


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # RAINFOREST_RESIN_fair_value = 10000  # Participant should calculate this value
        # RAINFOREST_RESIN_width = 2
        # RAINFOREST_RESIN_position_limit = 50
        #
        # KELP_make_width = 3.5
        # KELP_take_width = 1
        # KELP_position_limit = 50
        # KELP_timemspan = 10
        #
        # SQUID_INK_make_width = 3.5
        # SQUID_INK_take_width = 1
        # SQUID_INK_position_limit = 50
        # SQUID_INK_timemspan = 10

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            RAINFOREST_RESIN_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            RAINFOREST_RESIN_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    RAINFOREST_RESIN_position,
                )
            )
            RAINFOREST_RESIN_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    RAINFOREST_RESIN_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            RAINFOREST_RESIN_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                RAINFOREST_RESIN_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                RAINFOREST_RESIN_take_orders + RAINFOREST_RESIN_clear_orders + RAINFOREST_RESIN_make_orders
            )

        # if Product.RAINFOREST_RESIN in state.order_depths:
        #     RAINFOREST_RESIN_position = state.position[
        #         Product.RAINFOREST_RESIN] if Product.RAINFOREST_RESIN in state.position else 0
        #     RAINFOREST_RESIN_orders = self.RAINFOREST_RESIN_orders(state.order_depths[Product.RAINFOREST_RESIN],
        #                                                            RAINFOREST_RESIN_fair_value, RAINFOREST_RESIN_width,
        #                                                            RAINFOREST_RESIN_position,
        #                                                            RAINFOREST_RESIN_position_limit)
        #     result[Product.RAINFOREST_RESIN] = RAINFOREST_RESIN_orders

        # if Product.KELP in state.order_depths:
        #     KELP_position = state.position[Product.KELP] if Product.KELP in state.position else 0
        #     KELP_orders = self.KELP_orders(state.order_depths[Product.KELP], KELP_timemspan, KELP_make_width,
        #                                    KELP_take_width, KELP_position, KELP_position_limit)
        #     result[Product.KELP] = KELP_orders
        #
        # if Product.SQUID_INK in state.order_depths:
        #     SQUID_INK_position = state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
        #     SQUID_INK_orders = self.SQUID_INK_orders(state.order_depths[Product.SQUID_INK], SQUID_INK_timemspan, SQUID_INK_make_width,
        #                                    SQUID_INK_take_width, SQUID_INK_position, SQUID_INK_position_limit)
        #     result[Product.SQUID_INK] = SQUID_INK_orders

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            SQUID_INK_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    SQUID_INK_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_INK_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                SQUID_INK_fair_value,
                SQUID_INK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            result[Product.SQUID_INK] = (
                SQUID_INK_take_orders + SQUID_INK_clear_orders + SQUID_INK_make_orders
            )


        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData