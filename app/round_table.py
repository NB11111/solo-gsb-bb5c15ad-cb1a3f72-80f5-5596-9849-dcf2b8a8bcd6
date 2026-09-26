"""观测计数交叉表的受控舍入：每格归到原值相邻的 base 倍数并精确匹配目标行列合计。

直接公开逐格精确计数会泄露交叉维度上的细节分布。本模块把 1～20 行、
1～20 列的非负整数计数矩阵的每格舍入到原值向下或向上的 base 倍数
（整倍数格保持原值），使舍入后每行、每列的合计精确等于指定的目标值，
并在所有可行方案中最小化逐格绝对改变量之和（同成本时任选其一）。

每格只有“向下 / 向上”两种取值，因此问题等价于在“非整倍数格”构成的
行×列二部图上选择若干格向上取整：第 i 行被选中的格数必须为
``(row_totals[i] - 行内向下舍入和) / base``，列同理。每格的选择费用为
“向上与向下的改变量之差”（可为负，余数超过 base 一半时向上更便宜）。
这是标准的最小费用最大流问题，二部关联矩阵的全单模性保证最优流即
整数最优解；单侧缺口越界或可行流不存在时返回 422。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .anonymizer import unprocessable

# 矩阵行数与列数的上限。
MAX_DIMENSION = 20


def prepare_counts(raw_counts: Any) -> list[list[int]]:
    """校验计数矩阵，返回非负整数矩阵（1～20 行、1～20 列的矩形）。

    每格必须是非负整数（拒绝布尔、浮点与字符串）；所有行必须等长。
    """
    if not isinstance(raw_counts, list) or not raw_counts:
        raise unprocessable("counts 必须是至少包含 1 行的非空数组")
    if len(raw_counts) > MAX_DIMENSION:
        raise unprocessable(f"counts 最多包含 {MAX_DIMENSION} 行")

    matrix: list[list[int]] = []
    width: int | None = None
    for index, raw_row in enumerate(raw_counts):
        label = f"counts 第 {index + 1} 行"
        if not isinstance(raw_row, list) or not raw_row:
            raise unprocessable(f"{label}必须是至少包含 1 个计数的非空数组")
        if len(raw_row) > MAX_DIMENSION:
            raise unprocessable(f"{label}最多包含 {MAX_DIMENSION} 列")
        if width is None:
            width = len(raw_row)
        elif len(raw_row) != width:
            raise unprocessable(
                f"{label}的列数 {len(raw_row)} 与首行的 {width} 列不一致"
            )
        row: list[int] = []
        for value in raw_row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise unprocessable(f"{label}的每格计数必须是整数")
            if value < 0:
                raise unprocessable(f"{label}的每格计数必须是非负整数（>= 0）")
            row.append(value)
        matrix.append(row)
    return matrix


def prepare_totals(
    raw_totals: Any,
    expected_length: int,
    base: int,
    name: str,
    dimension: str,
) -> list[int]:
    """校验一侧目标合计：长度匹配的非负整数数组，每项都是 base 的倍数。"""
    if not isinstance(raw_totals, list):
        raise unprocessable(f"{name} 必须是数组")
    if len(raw_totals) != expected_length:
        raise unprocessable(
            f"{name} 的长度 {len(raw_totals)} 必须等于 counts 的{dimension}数 "
            f"{expected_length}"
        )

    totals: list[int] = []
    for value in raw_totals:
        if isinstance(value, bool) or not isinstance(value, int):
            raise unprocessable(f"{name} 的每项必须是整数")
        if value < 0:
            raise unprocessable(f"{name} 的每项必须是非负整数（>= 0）")
        if value % base != 0:
            raise unprocessable(
                f"{name} 的每项必须是 base={base} 的非负整数倍"
            )
        totals.append(value)
    return totals


def solve_rounding(
    counts: list[list[int]],
    base: int,
    row_totals: list[int],
    column_totals: list[int],
) -> tuple[list[list[int]], int]:
    """返回 (满足全部目标行列合计的最优舍入矩阵, 逐格绝对改变量之和)。

    每格只取原值向下或向上的 base 倍数（整倍数格保持原值）；在所有可行
    方案中最小化逐格绝对改变量之和，同成本时任选其一。不存在可行解时
    抛 422。
    """
    rows = len(counts)
    cols = len(counts[0])

    # 每格：向下舍入值、是否非整倍数（才可向上）、向上相对向下的费用差。
    lows: list[list[int]] = []
    flexible: list[list[bool]] = []
    weights: list[list[int]] = []
    for row in counts:
        low_row: list[int] = []
        flex_row: list[bool] = []
        weight_row: list[int] = []
        for value in row:
            remainder = value % base
            low_row.append(value - remainder)
            flex_row.append(remainder != 0)
            # 向上的改变量 (base - remainder) 减去向下的改变量 remainder。
            weight_row.append(base - 2 * remainder)
        lows.append(low_row)
        flexible.append(flex_row)
        weights.append(weight_row)

    row_ups = [
        _demand_ups(
            row_totals[i] - sum(lows[i]),
            sum(flexible[i]),
            base,
            row_totals[i],
            f"第 {i + 1} 行",
        )
        for i in range(rows)
    ]
    col_ups = [
        _demand_ups(
            column_totals[j] - sum(lows[i][j] for i in range(rows)),
            sum(flexible[i][j] for i in range(rows)),
            base,
            column_totals[j],
            f"第 {j + 1} 列",
        )
        for j in range(cols)
    ]

    chosen = _select_ups(row_ups, col_ups, flexible, weights)
    if chosen is None:
        raise unprocessable(
            "不存在满足全部目标行列合计的舍入方案："
            "单侧合计均在可取范围内，但行列约束无法同时满足"
        )

    rounded = [row[:] for row in lows]
    for i, j in chosen:
        rounded[i][j] += base
    total_change = sum(
        abs(rounded[i][j] - counts[i][j])
        for i in range(rows)
        for j in range(cols)
    )
    return rounded, total_change


def _demand_ups(
    deficit: int, capacity: int, base: int, target: int, label: str
) -> int:
    """把一侧的合计缺口换算为需要向上取整的格数，并校验该侧自身可行。

    ``deficit`` 必为 base 的倍数（目标合计与向下舍入值都是 base 的倍数），
    因此需要向上取整的格数为 ``deficit / base``，必须落在
    ``[0, 该侧非整倍数格数]`` 内。
    """
    need = deficit // base
    if need < 0:
        raise unprocessable(
            f"{label}的目标合计 {target} 小于全部向下舍入后的合计 "
            f"{target - deficit}，不存在可行解"
        )
    if need > capacity:
        raise unprocessable(
            f"{label}的目标合计 {target} 大于全部向上舍入后的合计 "
            f"{target - deficit + capacity * base}，不存在可行解"
        )
    return need


def _select_ups(
    row_ups: list[int],
    col_ups: list[int],
    flexible: list[list[bool]],
    weights: list[list[int]],
) -> set[tuple[int, int]] | None:
    """最小费用最大流选出向上取整的格集合；不存在可行流时返回 None。

    网络：源 -> 每行（容量 = 该行需向上的格数，费用 0），
    行 -> 列（每非整倍数格一条，容量 1，费用 = 向上相对向下的费用差），
    列 -> 汇（容量 = 该列需向上的格数，费用 0）。流量达到
    ``sum(row_ups)`` 即存在可行解，此时最小费用对应最小的总改变量。
    行列两侧需求之和相等（两侧目标总和相等），增广次数不超过格数。
    """
    rows = len(row_ups)
    cols = len(col_ups)
    source = rows + cols
    sink = source + 1
    # 邻接表：每条边为 [终点, 剩余容量, 单位费用, 反向边下标]。
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(frm: int, to: int, capacity: int, cost: int) -> None:
        graph[frm].append([to, capacity, cost, len(graph[to])])
        graph[to].append([frm, 0, -cost, len(graph[frm]) - 1])

    for i, need in enumerate(row_ups):
        if need:
            add_edge(source, i, need, 0)
    for j, need in enumerate(col_ups):
        if need:
            add_edge(rows + j, sink, need, 0)
    # 记录每个非整倍数格对应的正向边，供求解后读取“是否被选中”。
    cell_edge: dict[tuple[int, int], int] = {}
    for i in range(rows):
        for j in range(cols):
            if flexible[i][j]:
                cell_edge[(i, j)] = len(graph[i])
                add_edge(i, rows + j, 1, weights[i][j])

    # 逐最短路增广（SPFA 处理可能为负的费用）；每轮至少饱和一条
    # 容量为 1 的行->列边，因此轮数有限。
    total = sum(row_ups)
    flow = 0
    while flow < total:
        dist = [float("inf")] * len(graph)
        prev_node = [-1] * len(graph)
        prev_edge = [-1] * len(graph)
        in_queue = [False] * len(graph)
        dist[source] = 0
        queue: deque[int] = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge[1] > 0 and dist[node] + edge[2] < dist[edge[0]]:
                    dist[edge[0]] = dist[node] + edge[2]
                    prev_node[edge[0]] = node
                    prev_edge[edge[0]] = edge_index
                    if not in_queue[edge[0]]:
                        queue.append(edge[0])
                        in_queue[edge[0]] = True
        if dist[sink] == float("inf"):
            break

        # 沿最短路按瓶颈容量增广。
        pushed = total - flow
        node = sink
        while node != source:
            edge = graph[prev_node[node]][prev_edge[node]]
            pushed = min(pushed, edge[1])
            node = prev_node[node]
        flow += pushed
        node = sink
        while node != source:
            edge = graph[prev_node[node]][prev_edge[node]]
            edge[1] -= pushed
            graph[node][edge[3]][1] += pushed
            node = prev_node[node]

    if flow < total:
        return None
    # 正向边容量被耗尽的格即被选中向上取整。
    return {
        (i, j)
        for (i, j), edge_index in cell_edge.items()
        if graph[i][edge_index][1] == 0
    }
