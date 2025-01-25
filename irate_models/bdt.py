#!/usr/bin/env python3
import numpy as np
import math
from scipy import optimize as opt
import time


class Node:
    def __init__(self, tenor: int, id: int, left, right, rate: float):
        self.tenor = tenor
        self.id = id
        self.left = left
        self.right = right
        self.rate = rate


class BDT:
    def __init__(self, input_quotes: dict[int, list], max_tenor: int):
        self.input_quotes = input_quotes
        self.max_tenor = max_tenor
        self.p = 0.5  # Probability of up or down movement
        self.tree: list[list[Node]] = []
        try:
            self.input_rates = {key: value[0] for key, value in zip(self.input_quotes.keys(), self.input_quotes.values())}
            self.input_zcb_prices = {key: self.px0_zcb(value[0], key) for key, value in zip(self.input_quotes.keys(), self.input_quotes.values())}
            self.input_vols = {key: value[1] for key, value in zip(self.input_quotes.keys(), self.input_quotes.values())}
        except ValueError as e:
            raise ValueError(f"Invalid input quotes! {e}")

        print("Input Term Structure:")
        print("Tenor\tRate(%)\tZCB_Px\tVolatility(%)")
        for tenor, data, px in zip(self.input_quotes.keys(), self.input_quotes.values(), self.input_zcb_prices.values()):
            print(f"{tenor}\t{data[0] * 100}\t{round(px, 2)}\t{data[1] * 100}")

        print("Calibrating BDT Model...")
        start_time = time.time()
        try:
            self.build_tree()
        except Exception as e:
            raise Exception(f"Failed to build BDT model! {e}")
        end_time = time.time()
        calibration_time = (end_time - start_time) * 1000
        print(f"Calibration completed in {calibration_time:.2f} ms")

    def build_tree(self):
        for t in range(0, self.max_tenor):
            nodes_at_t = [Node(tenor=t, id=i, left=None, right=None, rate=self.input_rates[t + 1]) for i in range(0, t + 1)]
            if t == 0:
                nodes_at_t[0].rate = self.input_rates[1]
                nodes_at_t[0].previous = None
                self.tree.append(nodes_at_t)
                continue
            else:
                nodes_at_t_1 = self.tree[-1]
                guess_at_t = self.input_rates[t + 1]

                # Link up nodes
                for i, node in enumerate(nodes_at_t_1):
                    node.left = nodes_at_t[i]
                    node.right = nodes_at_t[i + 1]

            self.tree.append(nodes_at_t)

            # Fit nodes to input term structure using scipy
            # Constraint is rates must be positive
            res = opt.root(self.obj_func_t, guess_at_t, args=(t, self.input_vols[t + 1], self.input_zcb_prices[t + 1]), method='lm')
            if res.success:
                print(f"Calibration successful at t={t}!")
                guess = res.x[0]
                for i, node in enumerate(self.tree[t]):
                    node.rate = guess * math.exp((i - 1) * 2 * self.input_vols[t + 1])
            else:
                raise Exception(f"Calibration failed at t={t}!")

    def obj_func_t(self, guess: float, t: int, vol: float, px_t0: float):
        if guess < 0:
            return 1000

        num_nodes = t + 1
        guesses = [guess * math.exp((i - 1) * 2 * vol) for i in range(0, num_nodes)]

        # Assign guesses to nodes
        for i, node in enumerate(self.tree[t]):
            node.rate = guesses[i]

        # Calculate ZCB prices at T=0
        start_node = self.tree[0][0]
        zcb_t0 = self.px_at_node(start_node)

        # Calculate error
        constraint = px_t0 - zcb_t0
        return constraint

    def px_at_node(self, node: Node):
        if node.left is None:
            df = self.discount_factor(node.rate, 1)
            return 100 * df
        else:
            left_node = node.left
            right_node = node.right
            px = self.px_at_node(left_node) * self.p + self.px_at_node(right_node) * (1 - self.p)
            df = self.discount_factor(node.rate, 1)
            return px * df

    @staticmethod
    def discount_factor(rate: float, tenor: int):
        return np.power(1 + rate, -tenor)

    @staticmethod
    def px0_zcb(yld: float, tenor: int):
        return np.power(1 + yld, -tenor) * 100

    def print_tree(self):
        for i, nodes in enumerate(self.tree):
            print(f"Time: {i}")
            for node in nodes:
                print(f"Node: {node.id} Rate: {round(node.rate * 100, 2)}")


if __name__ == "__main__":
    input_dict = {1: [0.10, 0.2], 2: [0.11, 0.19], 3: [0.12, 0.18], 4: [0.125, 0.17], 5: [0.13, 0.16]}
    bdt = BDT(input_dict, 5)
    bdt.print_tree()
