import math
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple
import numpy as np


# 公共组件
class TSPBase:
    def __init__(self, cities):
        self.cities = cities
        self.n = len(cities)
        self.distance_matrix = self._calc_distance_matrix()
        self.best_solution = None
        self.best_cost = float('inf')
        self.history = []

    def _calc_distance_matrix(self):
        matrix = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    matrix[i][j] = math.sqrt(dx ** 2 + dy ** 2)
        return matrix

    def evaluate(self, solution):
        return sum(self.distance_matrix[solution[i]][solution[(i + 1) % self.n]] for i in range(self.n))


# 禁忌搜索算法
class TSPSolver(TSPBase):
    def __init__(self, cities, max_iter=1000, tabu_length=30, candidate_size=50):
        super().__init__(cities)
        self.max_iter = max_iter
        self.tabu_length = tabu_length
        self.candidate_size = candidate_size
        self.tabu_list = deque(maxlen=tabu_length)

    def optimize(self):
        current = random.sample(range(self.n), self.n)
        self.best_solution = current.copy()
        self.best_cost = self.evaluate(current)

        for iteration in range(self.max_iter):
            # 生成候选解
            candidates = []
            for _ in range(self.candidate_size):
                if random.random() < 0.7:  # 交换
                    i, j = random.sample(range(self.n), 2)
                    new = current.copy()
                    new[i], new[j] = new[j], new[i]
                else:  # 2-opt
                    i, j = sorted(random.sample(range(self.n), 2))
                    new = current[:i] + current[i:j][::-1] + current[j:]
                candidates.append(new)

            # 评估候选解
            best_candidate = None
            best_cost = float('inf')
            for sol in candidates:
                cost = self.evaluate(sol)
                move = tuple(sorted((current.index(sol[0]), current.index(sol[1]))))
                if (move not in self.tabu_list) or (cost < self.best_cost):
                    if cost < best_cost:
                        best_candidate = sol
                        best_cost = cost

            # 更新状态
            if best_candidate:
                current = best_candidate
                if best_cost < self.best_cost:
                    self.best_solution = current.copy()
                    self.best_cost = best_cost
                self.tabu_list.append(move)

            self.history.append(self.best_cost)
            if iteration % 100 == 0:
                print(f"TS Iteration {iteration}: {self.best_cost:.2f}")


# 模拟退火算法
class SimulatedAnnealing(TSPBase):
    def __init__(self, cities, max_iter=1000, temp=10000, cooling=0.995):
        super().__init__(cities)
        self.max_iter = max_iter
        self.temp = temp
        self.cooling = cooling

    def optimize(self):
        current = random.sample(range(self.n), self.n)
        current_cost = self.evaluate(current)
        self.best_solution = current.copy()
        self.best_cost = current_cost

        for iteration in range(self.max_iter):
            # 生成邻域解
            i, j = random.sample(range(self.n), 2)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_cost = self.evaluate(neighbor)

            # 接受准则
            if neighbor_cost < current_cost or \
                    random.random() < math.exp((current_cost - neighbor_cost) / self.temp):
                current, current_cost = neighbor, neighbor_cost
                if current_cost < self.best_cost:
                    self.best_solution = current.copy()
                    self.best_cost = current_cost

            self.temp *= self.cooling
            self.history.append(self.best_cost)
            if iteration % 100 == 0:
                print(f"SA Iteration {iteration}: {self.best_cost:.2f}")


# 遗传算法
class GeneticAlgorithm(TSPBase):
    def __init__(self, cities, max_iter=500, pop_size=100, elite_size=20, mutation_rate=0.01):
        super().__init__(cities)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    def optimize(self):
        population = [random.sample(range(self.n), self.n) for _ in range(self.pop_size)]

        for iteration in range(self.max_iter):
            # 评估种群
            fitness = [1 / self.evaluate(ind) for ind in population]
            best_idx = np.argmax(fitness)
            current_best = population[best_idx]
            current_cost = 1 / fitness[best_idx]

            if current_cost < self.best_cost:
                self.best_solution = current_best.copy()
                self.best_cost = current_cost

            # 选择精英
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            elites = [population[i] for i in elite_indices]

            # 交叉和变异
            new_population = elites.copy()
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.choices(population, weights=fitness, k=2)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    i, j = random.sample(range(self.n), 2)
                    child[i], child[j] = child[j], child[i]
                new_population.append(child)

            population = new_population
            self.history.append(self.best_cost)
            if iteration % 50 == 0:
                print(f"GA Generation {iteration}: {self.best_cost:.2f}")

    def _crossover(self, parent1, parent2):
        # 顺序交叉(OX)
        start, end = sorted(random.sample(range(self.n), 2))
        child = [None] * self.n
        child[start:end] = parent1[start:end]

        ptr = end
        for gene in parent2:
            if gene not in child:
                if ptr >= self.n: ptr = 0
                child[ptr] = gene
                ptr += 1
        return child


# 可视化函数
def plot_comparison(algorithms, names):
    plt.figure(figsize=(12, 6))
    colors = ['#2c7bb6', '#d7191c', '#fdae61']

    for algo, name, color in zip(algorithms, names, colors):
        plt.plot(algo.history, label=f'{name} (Final: {algo.best_cost:.2f})',
                 color=color, linewidth=2)

    plt.title('Algorithm Comparison (50-City TSP)', fontsize=16)
    plt.xlabel('Iteration/Generation', fontsize=12)
    plt.ylabel('Best Tour Length', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"algorithm_convergence saved to algorithm_comparison.png")
    plt.show()


def plot_routes(cities, solutions, names):
    plt.figure(figsize=(15, 5))
    colors = ['#2c7bb6', '#d7191c', '#fdae61']

    for i, (sol, name, color) in enumerate(zip(solutions, names, colors), 1):
        plt.subplot(1, 3, i)
        x = [c[0] for c in cities]
        y = [c[1] for c in cities]

        plt.scatter(x, y, c='gray', s=30, alpha=0.7)
        route_x = [x[i] for i in sol] + [x[sol[0]]]
        route_y = [y[i] for i in sol] + [y[sol[0]]]
        plt.plot(route_x, route_y, color=color, linewidth=1.5)

        plt.title(f'{name} (Length: {algorithms[i - 1].best_cost:.2f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

    plt.tight_layout()
    plt.savefig('route_comparison.png', dpi=300)
    plt.show()


# 主程序
if __name__ == "__main__":
    cities = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
    (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
    (180, 100), (60, 80), (120, 80), (180, 60), (20, 40),
    (100, 40), (200, 40), (20, 20), (60, 20), (160, 20),
    (50, 150), (110, 150), (170, 150), (70, 130), (130, 130),
    (190, 130), (30, 110), (90, 110), (150, 110), (10, 90),
    (80, 90), (160, 90), (40, 70), (120, 70), (200, 70),
    (20, 50), (100, 50), (180, 50), (60, 30), (140, 30),
    (90, 170), (170, 170), (30, 150), (150, 150), (50, 130),
    (130, 130), (10, 110), (70, 110), (190, 110), (110, 90)]  # 使用相同的50城市数据

    # 运行算法
    ts = TSPSolver(cities, max_iter=1000)
    sa = SimulatedAnnealing(cities, max_iter=1000)
    ga = GeneticAlgorithm(cities, max_iter=1000)

    print("Running Tabu Search...")
    ts.optimize()
    print("\nRunning Simulated Annealing...")
    sa.optimize()
    print("\nRunning Genetic Algorithm...")
    ga.optimize()

    # 可视化
    algorithms = [ts, sa, ga]
    names = ['Tabu Search', 'Simulated Annealing', 'Genetic Algorithm']

    plot_comparison(algorithms, names)
    plot_routes(cities, [a.best_solution for a in algorithms], names)