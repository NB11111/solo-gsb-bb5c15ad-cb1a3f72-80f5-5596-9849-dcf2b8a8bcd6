"""观测计数交叉表受控舍入接口 POST /api/v1/round-count-table 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/round-count-table"


def payload(
    counts=None,
    base: int = 5,
    row_totals=None,
    column_totals=None,
) -> dict:
    """默认请求：README 中的 3×3 示例（唯一最优解，总改变量 16）。"""
    return {
        "counts": counts if counts is not None else [[3, 7, 12], [5, 2, 8], [11, 6, 1]],
        "base": base,
        "row_totals": row_totals if row_totals is not None else [20, 15, 20],
        "column_totals": column_totals if column_totals is not None else [20, 15, 20],
    }


def brute_force_min_change(counts, base, row_totals, column_totals):
    """枚举每个非整倍数格的向下/向上选择，返回最小总改变量（不可行返回 None）。"""
    rows, cols = len(counts), len(counts[0])
    flex = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] % base != 0
    ]
    best = None
    for mask in range(1 << len(flex)):
        rounded = [[(v // base) * base for v in row] for row in counts]
        for bit, (i, j) in enumerate(flex):
            if mask >> bit & 1:
                rounded[i][j] += base
        if [sum(row) for row in rounded] != row_totals:
            continue
        if [sum(col) for col in zip(*rounded)] != column_totals:
            continue
        change = sum(
            abs(rounded[i][j] - counts[i][j])
            for i in range(rows)
            for j in range(cols)
        )
        best = change if best is None else min(best, change)
    return best


def assert_valid_rounding(body: dict, counts, base, row_totals, column_totals):
    """校验响应形状与舍入规则：每格只取向下/向上的 base 倍数，合计达标。"""
    assert set(body) == {
        "rounded_counts",
        "row_totals",
        "column_totals",
        "total_change",
        "base",
    }
    rounded = body["rounded_counts"]
    assert len(rounded) == len(counts)
    expected_change = 0
    for i, row in enumerate(counts):
        assert len(rounded[i]) == len(row)
        for j, value in enumerate(row):
            down = (value // base) * base
            up = down + (base if value % base else 0)
            assert rounded[i][j] in (down, up), (value, rounded[i][j])
            expected_change += abs(rounded[i][j] - value)
    assert body["row_totals"] == [sum(row) for row in rounded] == row_totals
    assert body["column_totals"] == [sum(col) for col in zip(*rounded)] == column_totals
    assert body["total_change"] == expected_change
    assert body["base"] == base


# ---------------------------------------------------------------- 基本功能

def test_readme_example_unique_optimum():
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()
    # 唯一最优解：向上的格为 (0,0)、(1,2)、(2,1)，总改变量 16。
    assert body["rounded_counts"] == [[5, 5, 10], [5, 0, 10], [10, 10, 0]]
    assert body["row_totals"] == [20, 15, 20]
    assert body["column_totals"] == [20, 15, 20]
    assert body["total_change"] == 16
    assert body["base"] == 5


def test_exact_multiples_stay_unchanged():
    counts = [[10, 5], [0, 15]]
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=counts,
            row_totals=[15, 15],
            column_totals=[10, 20],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == counts
    assert body["total_change"] == 0


def test_base_one_keeps_every_value():
    counts = [[3, 8], [6, 1]]
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=counts,
            base=1,
            row_totals=[11, 7],
            column_totals=[9, 9],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == counts
    assert body["total_change"] == 0


def test_single_cell_round_up_and_down():
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[7]], row_totals=[10], column_totals=[10]),
    )
    assert response.status_code == 200
    assert response.json()["rounded_counts"] == [[10]]
    assert response.json()["total_change"] == 3

    response = client.post(
        ENDPOINT,
        json=payload(counts=[[7]], row_totals=[5], column_totals=[5]),
    )
    assert response.status_code == 200
    assert response.json()["rounded_counts"] == [[5]]
    assert response.json()["total_change"] == 2


def test_tie_returns_some_optimal_solution():
    # 四格余数相同（base=2、余 1），任意满足合计的舍入成本都相同。
    counts = [[1, 1], [1, 1]]
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=counts,
            base=2,
            row_totals=[2, 2],
            column_totals=[2, 2],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert_valid_rounding(body, counts, 2, [2, 2], [2, 2])
    assert body["total_change"] == 4


def test_matches_brute_force_optimum():
    counts = [[4, 9, 1], [6, 3, 8], [2, 7, 5]]
    row_totals = [10, 10, 15]
    column_totals = [5, 15, 15]
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=counts,
            row_totals=row_totals,
            column_totals=column_totals,
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert_valid_rounding(body, counts, 5, row_totals, column_totals)
    assert body["total_change"] == brute_force_min_change(
        counts, 5, row_totals, column_totals
    )


def test_max_dimension_matrix():
    # 20×20 上限矩阵：目标取“全部向下舍入”的合计，必然可行。
    base = 3
    counts = [[(i * 20 + j) % 7 for j in range(20)] for i in range(20)]
    rounded_down = [[(v // base) * base for v in row] for row in counts]
    row_totals = [sum(row) for row in rounded_down]
    column_totals = [sum(col) for col in zip(*rounded_down)]
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=counts,
            base=base,
            row_totals=row_totals,
            column_totals=column_totals,
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert_valid_rounding(body, counts, base, row_totals, column_totals)
    # 最优解的改变量不会超过“全部向下”的方案。
    all_down_change = sum(v % base for row in counts for v in row)
    assert body["total_change"] <= all_down_change


def test_zero_matrix_with_zero_totals():
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[0, 0], [0, 0]], row_totals=[0, 0], column_totals=[0, 0]),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[0, 0], [0, 0]]
    assert body["total_change"] == 0


def test_root_lists_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["round_count_table"] == ENDPOINT


# ---------------------------------------------------------------- 不可行 422

def test_422_when_row_target_unreachable():
    # 第 1 行 [0, 0] 只能凑出 0，目标 5 不可达。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[0, 0]], row_totals=[5], column_totals=[5, 0]),
    )
    assert response.status_code == 422
    assert "不可达" in response.json()["detail"]


def test_422_when_column_target_unreachable():
    # 第 1 列只有 1 格（最大 5），目标 10 不可达；行目标 10 本身可达。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[3, 3]], row_totals=[10], column_totals=[10, 0]),
    )
    assert response.status_code == 422
    assert "不可达" in response.json()["detail"]


def test_422_when_base_one_target_differs_from_original():
    # base=1 时所有值都是整倍数，目标合计必须等于原始合计。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[3]], base=1, row_totals=[4], column_totals=[4]),
    )
    assert response.status_code == 422


def test_422_when_joint_constraints_infeasible():
    # 单行单列各自可达，但组合约束不可行：第 1 行唯一的可动格 (0,0)
    # 所在第 1 列不需要向上，第 2 列需要的向上又无人能提供。
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=[[1, 0], [0, 1]],
            base=2,
            row_totals=[2, 0],
            column_totals=[0, 2],
        ),
    )
    assert response.status_code == 422
    assert "不存在" in response.json()["detail"]


# ---------------------------------------------------------------- 参数 422

def test_422_on_counts_structure():
    bad_counts = [
        None,               # 缺失
        "not-a-list",       # 非数组
        [],                 # 空矩阵
        [[], []],           # 空行
        [[]],               # 单行空行
        [[1], "row"],       # 行不是数组
        [[1, 2], [3]],      # 非矩形
        [[1] * 21],         # 超过 20 列
        [[1]] * 21,         # 超过 20 行
    ]
    for counts in bad_counts:
        body = payload()
        if counts is not None:
            body["counts"] = counts
        else:
            del body["counts"]
        response = client.post(ENDPOINT, json=body)
        assert response.status_code == 422, counts


def test_422_on_invalid_cell_values():
    for bad_cell in (-1, True, 2.5, "3", None, [1], {"v": 1}):
        body = payload(counts=[[3, bad_cell], [5, 2]])
        body["row_totals"] = [10, 5]
        body["column_totals"] = [10, 5]
        response = client.post(ENDPOINT, json=body)
        assert response.status_code == 422, bad_cell


def test_422_on_invalid_base():
    for bad_base in (0, -5, True, 2.5, "5", None):
        body = payload(base=bad_base)
        response = client.post(ENDPOINT, json=body)
        assert response.status_code == 422, bad_base


def test_422_on_invalid_row_totals():
    base_body = payload()
    variants = [
        [20, 15],             # 长度不足
        [20, 15, 20, 5],      # 长度过长
        [20, -5, 20],         # 负数
        [20, 16, 19],         # 非 base 倍数
        [20, True, 20],       # 布尔
        [20, "15", 20],       # 字符串
        "not-a-list",         # 非数组
    ]
    for row_totals in variants:
        body = dict(base_body)
        body["row_totals"] = row_totals
        response = client.post(ENDPOINT, json=body)
        assert response.status_code == 422, row_totals


def test_422_on_invalid_column_totals():
    base_body = payload()
    variants = [
        [20, 15],             # 长度不足
        [20, 15, 20, 5],      # 长度过长
        [-5, 15, 20],         # 负数
        [21, 15, 20],         # 非 base 倍数
        [20, 2.5, 20],        # 浮点
        [20, None, 20],       # 空值
        {"0": 20},            # 非数组
    ]
    for column_totals in variants:
        body = dict(base_body)
        body["column_totals"] = column_totals
        response = client.post(ENDPOINT, json=body)
        assert response.status_code == 422, column_totals


def test_422_when_side_sums_differ():
    body = payload(row_totals=[20, 15, 20], column_totals=[20, 15, 25])
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 422
    assert "相等" in response.json()["detail"]


def test_422_on_missing_or_extra_fields():
    for key in ("counts", "base", "row_totals", "column_totals"):
        body = payload()
        del body[key]
        response = client.post(ENDPOINT, json=body)
        assert response.status_code == 422, key

    body = payload()
    body["unexpected"] = 1
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 422


# ---------------------------------------------------------------- 持久化

def test_no_table_written_to_disk(tmp_path, monkeypatch):
    """处理请求前后工作目录下不应出现任何数据文件（服务纯内存）。"""
    monkeypatch.chdir(tmp_path)
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    assert list(tmp_path.rglob("*")) == []
