"""观测计数交叉表受控舍入：逐格舍入到 base 倍数并满足目标行列合计。

为降低公开发布的观测计数交叉表（如“保护站 × 物种”计数矩阵）的精度，
调用方给出非负整数矩阵 counts、正整数基数 base 与目标行/列合计。每格只能
取原值向下或向上到 base 倍数的值，整倍数格保持原值：对
``counts[i][j] = base*q + r``（``0 <= r < base``），``r = 0`` 时值不变，
否则取 ``base*q``（向下）或 ``base*(q+1)``（向上）。

以 base 为单位，行 i 需要“向上”的格数
``R[i] = row_totals[i]/base - 该行 q 之和``，列 j 需要
``C[j] = column_totals[j]/base - 该列 q 之和``。每格“向下”的绝对改变量为
r、“向上”为 base - r，因此总改变量 = 全部 r 之和（常数）+ 各向上格的
(base - 2r)，问题化为如下最小费用流：

- 源点 -> 行 i：容量 R[i]，费用 0；
- 行 i -> 列 j（仅 r > 0 的格）：容量 1，费用 base - 2r；
- 列 j -> 汇点：容量 C[j]，费用 0。

流量为 R 之和的最小费用流即给出逐格绝对改变量之和最小的舍入矩阵；同成本
的最优解可能不止一个，返回其中任一。不存在可行流时返回 422。

全部处理在请求内存中完成，不记录、不落盘原始矩阵。
"""

from __future__ import annotations

import heapq
import math
from typing import Any

from .anonymizer import unprocessable

# counts 矩阵行数与列数的上限。
MAX_DIMENSION = 20


def prepare_counts(raw_counts: Any) -> list[list[int]]:
    """校验计数矩阵，返回整型二维列表（保持输入行列顺序）。

    必须是 1～20 行、1～20 列的矩形数组；每格为非负整数（拒绝布尔、
    浮点与字符串）。
    """
    if not isinstance(raw_counts, list) or not raw_counts:
        raise unprocessable(
            f"counts 必须是非空的二维数组（1 到 {MAX_DIMENSION} 行）"
        )
    if len(raw_counts) > MAX_DIMENSION:
        raise unprocessable(f"counts 最多 {MAX_DIMENSION} 行")

    counts: list[list[int]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(raw_counts):
        label = f"counts 第 {row_index + 1} 行"
        if not isinstance(raw_row, list) or not raw_row:
            raise unprocessable(
                f"{label}必须是非空的数组（1 到 {MAX_DIMENSION} 列）"
            )
        if len(raw_row) > MAX_DIMENSION:
            raise unprocessable(f"{label}最多 {MAX_DIMENSION} 列")
        if width is None:
            width = len(raw_row)
        elif len(raw_row) != width:
            raise unprocessable(
                f"{label}为 {len(raw_row)} 列，与第 1 行的 {width} 列不一致，"
                "counts 必须是矩形矩阵"
            )

        row: list[int] = []
        for column_index, value in enumerate(raw_row):
            cell_label = f"{label}第 {column_index + 1} 列"
            if isinstance(value, bool) or not isinstance(value, int):
                raise unprocessable(f"{cell_label}必须是整数")
            if value < 0:
                raise unprocessable(f"{cell_label}必须是非负整数（>= 0）")
            row.append(value)
        counts.append(row)

    return counts


def prepare_totals(
    raw_row_totals: Any,
    raw_column_totals: Any,
    rows: int,
    columns: int,
    base: int,
) -> tuple[list[int], list[int]]:
    """校验目标行列合计，返回 (row_totals, column_totals)。

    长度必须分别等于 counts 的行数与列数；每项都是非负整数且为 base 的
    整数倍；两侧总和必须相等。
    """
    row_totals = _prepare_side_totals(raw_row_totals, rows, "row_totals", "行", base)
    column_totals = _prepare_side_totals(
        raw_column_totals, columns, "column_totals", "列", base
    )

    row_sum = sum(row_totals)
    column_sum = sum(column_totals)
    if row_sum != column_sum:
        raise unprocessable(
            f"row_totals 之和（{row_sum}）与 column_totals 之和（{column_sum}）"
            "必须相等"
        )
    return row_totals, column_totals


def _prepare_side_totals(
    raw_totals: Any, expected: int, name: str, side: str, base: int
) -> list[int]:
    """校验一侧（行或列）的目标合计数组。"""
    if not isinstance(raw_totals, list):
        raise unprocessable(f"{name} 必须是数组")
    if len(raw_totals) != expected:
        raise unprocessable(
            f"{name} 的长度必须等于 counts 的{side}数（{expected}）"
        )

    totals: list[int] = []
    for index, value in enumerate(raw_totals):
        label = f"{name} 第 {index + 1} 项"
        if isinstance(value, bool) or not isinstance(value, int):
            raise unprocessable(f"{label}必须是整数")
        if value < 0:
            raise unprocessable(f"{label}必须是非负整数（>= 0）")
        if value % base != 0:
            raise unprocessable(f"{label}必须是 base={base} 的整数倍")
        totals.append(value)
    return totals


def solve_rounding(
    counts: list[list[int]],
    base: int,
    row_totals: list[int],
    column_totals: list[int],
) -> tuple[list[list[int]], int]:
    """求满足目标行列合计且逐格绝对改变量之和最小的舍入矩阵。

    返回 (舍入矩阵, 总改变量)；不存在可行解时抛出 422。
    """
    rows = len(counts)
    columns = len(counts[0])

    # counts[i][j] = base*q + r（0 <= r < base）：向下取 base*q，向上取
    # base*(q+1)；r = 0 的整倍数格保持原值，不参与“向上”分配。
    quotients = [[value // base for value in row] for row in counts]
    remainders = [[value % base for value in row] for row in counts]

    # 以 base 为单位：行 i / 列 j 分别还需要多少个“向上”的格。
    row_up = [row_totals[i] // base - sum(quotients[i]) for i in range(rows)]
    column_up = [
        column_totals[j] // base - sum(quotients[i][j] for i in range(rows))
        for j in range(columns)
    ]

    # 单行/单列可达性预检（同时保证下面不会出现负容量弧）：行 i 的向上格数
    # 只能在 0 到该行非整倍数格数之间，列同理。
    for i in range(rows):
        slots = sum(1 for j in range(columns) if remainders[i][j] > 0)
        if not 0 <= row_up[i] <= slots:
            low = base * sum(quotients[i])
            raise unprocessable(
                f"第 {i + 1} 行的目标合计 {row_totals[i]} 不可达："
                f"该行各格只能凑出 {low} 到 {low + base * slots} 之间的 base 倍数"
            )
    for j in range(columns):
        slots = sum(1 for i in range(rows) if remainders[i][j] > 0)
        if not 0 <= column_up[j] <= slots:
            low = base * sum(quotients[i][j] for i in range(rows))
            raise unprocessable(
                f"第 {j + 1} 列的目标合计 {column_totals[j]} 不可达："
                f"该列各格只能凑出 {low} 到 {low + base * slots} 之间的 base 倍数"
            )

    # 最小费用流：源点 -> 行（容量 row_up）、行 -> 列（仅 r>0 的格，容量 1，
    # 费用 base-2r，即该格“向上”相对“向下”的改变量差）、列 -> 汇点。
    source = 0
    sink = rows + columns + 1
    arcs: list[tuple[int, int, int, int]] = []
    for i in range(rows):
        arcs.append((source, 1 + i, row_up[i], 0))
    cell_arc: dict[tuple[int, int], int] = {}
    for i in range(rows):
        for j in range(columns):
            if remainders[i][j] > 0:
                cell_arc[(i, j)] = len(arcs)
                arcs.append((1 + i, 1 + rows + j, 1, base - 2 * remainders[i][j]))
    for j in range(columns):
        arcs.append((1 + rows + j, sink, column_up[j], 0))

    flows = _min_cost_flow(sink + 1, arcs, source, sink, sum(row_up))
    if flows is None:
        raise unprocessable("不存在满足全部目标行列合计的舍入方案")

    rounded: list[list[int]] = []
    for i in range(rows):
        rounded_row: list[int] = []
        for j in range(columns):
            arc_index = cell_arc.get((i, j))
            up = flows[arc_index] if arc_index is not None else 0
            rounded_row.append(base * (quotients[i][j] + up))
        rounded.append(rounded_row)

    total_change = sum(
        abs(rounded[i][j] - counts[i][j])
        for i in range(rows)
        for j in range(columns)
    )
    return rounded, total_change


def _min_cost_flow(
    num_nodes: int,
    arcs: list[tuple[int, int, int, int]],
    source: int,
    sink: int,
    required: int,
) -> list[int] | None:
    """返回流量为 required 的最小费用流中每条弧的流量；不可行时返回 None。

    弧为 (起点, 终点, 容量, 单位费用)，费用允许为负。连续最短路算法：
    先以 Bellman-Ford 求初始势（原始网络无环，必无负环），之后每轮用
    Dijkstra 沿约化费用最短路增广；容量均为整数，流量必为整数，结果在
    所有可行流中费用最小。
    """
    # 邻接表边：[终点, 反向边下标, 残量容量, 单位费用]。
    graph: list[list[list[int]]] = [[] for _ in range(num_nodes)]
    locations: list[tuple[int, int]] = []
    for start, end, capacity, cost in arcs:
        locations.append((start, len(graph[start])))
        graph[start].append([end, len(graph[end]), capacity, cost])
        graph[end].append([start, len(graph[start]) - 1, 0, -cost])

    # 初始势：距离全 0 起做 Bellman-Ford（等价于引入到各点零费用的虚拟
    # 源点），使所有残量边的约化费用非负，之后才能使用 Dijkstra。
    potential = [0] * num_nodes
    for _ in range(num_nodes - 1):
        updated = False
        for start in range(num_nodes):
            for edge in graph[start]:
                end, _, capacity, cost = edge
                if capacity > 0 and potential[start] + cost < potential[end]:
                    potential[end] = potential[start] + cost
                    updated = True
        if not updated:
            break

    flow = 0
    while flow < required:
        dist = [math.inf] * num_nodes
        prev_node = [-1] * num_nodes
        prev_edge = [-1] * num_nodes
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            d, start = heapq.heappop(heap)
            if d > dist[start]:
                continue
            for index, edge in enumerate(graph[start]):
                end, _, capacity, cost = edge
                if capacity <= 0:
                    continue
                candidate = d + cost + potential[start] - potential[end]
                if candidate < dist[end]:
                    dist[end] = candidate
                    prev_node[end] = start
                    prev_edge[end] = index
                    heapq.heappush(heap, (candidate, end))
        if math.isinf(dist[sink]):
            return None
        for node in range(num_nodes):
            if not math.isinf(dist[node]):
                potential[node] += dist[node]

        # 沿最短路增广瓶颈容量（格边容量为 1，通常一次只增广 1 单位）。
        add = required - flow
        node = sink
        while node != source:
            start = prev_node[node]
            add = min(add, graph[start][prev_edge[node]][2])
            node = start
        node = sink
        while node != source:
            start = prev_node[node]
            edge = graph[start][prev_edge[node]]
            edge[2] -= add
            graph[node][edge[1]][2] += add
            node = start
        flow += add

    return [
        capacity - graph[start][index][2]
        for (start, index), (_, _, capacity, _) in zip(locations, arcs)
    ]
