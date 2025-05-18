import math
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple


class TSPSolver:
    def __init__(self,
                 cities: List[Tuple[int, int]],
                 max_iter: int = 1000,
                 base_tabu_length: int = 50,
                 candidate_size: int = 50):

        # 数据初始化
        self.cities = cities
        self.n = len(cities)
        self.distance_matrix = self._calc_distance_matrix()

        # 算法参数
        self.max_iter = max_iter
        self.base_tabu_length = base_tabu_length
        self.candidate_size = candidate_size

        # 状态变量
        self.current_solution = None
        self.best_solution = None
        self.best_cost = float('inf')
        self.tabu_list = deque(maxlen=base_tabu_length)
        self.history = {'best_cost': [], 'current_cost': []}

        # 初始化解决方案
        self._initialize_solution()

    def _calc_distance_matrix(self) -> List[List[float]]:
        """计算距离矩阵"""
        matrix = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    matrix[i][j] = math.sqrt(dx ** 2 + dy ** 2)
        return matrix

    def _initialize_solution(self):
        """生成初始解"""
        random.seed(100)
        self.current_solution = list(range(self.n))
        random.shuffle(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_cost = self._evaluate(self.best_solution)

    def _dynamic_tabu_length(self) -> int:
        """动态禁忌长度策略"""
        current_iter = len(self.history['best_cost'])
        return int(self.base_tabu_length * (1 + math.log(current_iter + 1) / 10))

    def _generate_neighborhood(self) -> List[List[int]]:
        """混合邻域生成（交换操作+2-opt）"""
        candidates = []
        for _ in range(self.candidate_size):
            i, j = sorted(random.sample(range(self.n), 2))
            new_sol = self.current_solution.copy()
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
            candidates.append(new_sol)
        return candidates

    def _evaluate(self, solution: List[int]) -> float:
        """计算路径长度"""
        total = 0.0
        for i in range(self.n):
            total += self.distance_matrix[solution[i]][solution[(i + 1) % self.n]]
        return total

    def optimize(self):
        """执行优化主循环"""
        for iteration in range(self.max_iter):
            candidates = self._generate_neighborhood()

            best_candidate_cost = float('inf')
            best_candidate = None
            selected_move = None

            for sol in candidates:
                cost = self._evaluate(sol)
                # 移动表示为排序后的城市索引对
                move = tuple(sorted((
                    self.current_solution.index(sol[0]),
                    self.current_solution.index(sol[1]))
                )) if len(sol) > 1 else (0, 0)

                # 破禁条件判断
                if (move in self.tabu_list and cost < self.best_cost) \
                        or (move not in self.tabu_list):

                    if cost < best_candidate_cost:
                        best_candidate_cost = cost
                        best_candidate = sol
                        selected_move = move

            # 更新当前解
            if best_candidate is not None:
                self.current_solution = best_candidate
                current_cost = best_candidate_cost

                # 更新最优解
                if current_cost < self.best_cost:
                    self.best_solution = best_candidate.copy()
                    self.best_cost = current_cost

                # 更新禁忌表
                self.tabu_list.append(selected_move)
                self.tabu_list = deque(
                    list(self.tabu_list)[-self._dynamic_tabu_length():],
                    maxlen=self._dynamic_tabu_length()
                )

            # 记录历史数据
            self.history['best_cost'].append(self.best_cost)
            self.history['current_cost'].append(current_cost)

            # 打印进度
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best Cost = {self.best_cost:.2f}")

        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['best_cost'], label='Best Cost')
        plt.plot(self.history['current_cost'], label='Current Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Tour Length')
        plt.title('Convergence Curve')
        plt.legend()
        plt.savefig('convergence.png')
        plt.close()

        # 绘制收敛曲线（论文规格）
        plt.figure(figsize=(12, 8), dpi=300)  # 提高分辨率和尺寸
        plt.plot(self.history['best_cost'],
                 color='#2c7bb6',
                 linewidth=2.5,
                 linestyle='-',
                 label='Best Solution Length')

        plt.plot(self.history['current_cost'],
                 color='#d7191c',
                 linewidth=1.2,
                 linestyle='--',
                 alpha=0.7,
                 label='Current Solution Length')

        # 标注最优解
        min_idx = self.history['best_cost'].index(min(self.history['best_cost']))
        plt.scatter(min_idx, self.best_cost,
                    color='#fdae61',
                    s=120,
                    zorder=5,
                    label=f'Optimal Point ({self.best_cost:.2f})')

        # 图表格式设置
        plt.xlabel('Iteration', fontsize=14, fontweight='bold')
        plt.ylabel('Tour Length', fontsize=14, fontweight='bold')
        plt.title('Tabu Search Convergence Profile\n(50-City TSP Instance)',
                  fontsize=16,
                  pad=20)

        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12,
                   loc='upper right',
                   frameon=True,
                   shadow=True)

        # 坐标轴范围优化
        plt.xlim(0, len(self.history['best_cost']))
        buffer = 0.1 * (max(self.history['best_cost']) - self.best_cost)
        plt.ylim(self.best_cost - buffer, max(self.history['best_cost']) + buffer)

        # 保存多种格式
        plt.savefig('convergence.pdf', bbox_inches='tight')  # 矢量格式用于论文
        plt.savefig('convergence.png', dpi=300, bbox_inches='tight')  # 位图格式用于预览
        print(f"convergence saved to convergence.png/.pdg")
        plt.close()

        # 输出最终结果
        print(f"\nBest Tour Length: {self.best_cost:.2f}")
        print("Optimal Route:", self.best_solution)


def plot_optimal_route(cities, solution, save_path='optimal_route.png'):
    """在坐标图上绘制TSP最优路径

    Args:
        cities: 城市坐标列表 [(x1,y1), (x2,y2), ...]
        solution: 最优路径顺序 [idx1, idx2, ..., idxn]
        save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8), dpi=300)

    # 提取坐标
    x = [city[0] for city in cities]
    y = [city[1] for city in cities]

    # 绘制城市散点
    plt.scatter(x, y, c='red', s=100, edgecolors='black', zorder=10, label='Cities')

    # 添加城市标签
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi + 3, yi + 3, str(i), fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 绘制路径连线
    route_x = [x[i] for i in solution] + [x[solution[0]]]  # 闭合路径
    route_y = [y[i] for i in solution] + [y[solution[0]]]
    plt.plot(route_x, route_y,
             color='#2c7bb6',
             linewidth=1.5,
             linestyle='-',
             marker='o',
             markersize=6,
             markerfacecolor='yellow',
             markeredgecolor='black',
             label='Optimal Route')

    # 标注起点和终点
    start_x, start_y = x[solution[0]], y[solution[0]]
    plt.scatter(start_x, start_y,
                c='green', s=200,
                edgecolors='black',
                marker='*',
                zorder=11,
                label='Start/End')

    # 图表美化
    plt.title(f'TSP Optimal Route (Total Length: {solver.best_cost:.2f})',
              fontsize=16, pad=20)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Optimal route saved to {save_path}")

# 实验配置
if __name__ == "__main__":
    # 固定城市坐标（与论文实验一致）
    cities = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
    (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
    (180, 100), (60, 80), (120, 80), (180, 60), (20, 40),
    (100, 40), (200, 40), (20, 20), (60, 20), (160, 20),
    (50, 150), (110, 150), (170, 150), (70, 130), (130, 130),
    (190, 130), (30, 110), (90, 110), (150, 110), (10, 90),
    (80, 90), (160, 90), (40, 70), (120, 70), (200, 70),
    (20, 50), (100, 50), (180, 50), (60, 30), (140, 30),
    (90, 170), (170, 170), (30, 150), (150, 150), (50, 130),
    (130, 130), (10, 110), (70, 110), (190, 110), (110, 90)]  # 保持原有坐标数据
    # 初始化求解器
    solver = TSPSolver(
        cities=cities,
        max_iter=1000,
        base_tabu_length=int(math.sqrt(len(cities))),
        candidate_size=2 * len(cities)
    )

    # 执行优化
    solver.optimize()
    plot_optimal_route(cities, solver.best_solution)
