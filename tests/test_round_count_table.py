"""观测计数交叉表受控舍入接口 POST /api/v1/round-count-table 的端到端测试。"""

from __future__ import annotations

from itertools import combinations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/round-count-table"

# 基础样例：2×2 矩阵，base=10。
COUNTS = [[12, 7], [3, 9]]


def payload(
    counts=None,
    base: int = 10,
    row_totals=None,
    column_totals=None,
    **extra,
) -> dict:
    body = {
        "counts": COUNTS if counts is None else counts,
        "base": base,
        "row_totals": [20, 10] if row_totals is None else row_totals,
        "column_totals": [10, 20] if column_totals is None else column_totals,
    }
    body.update(extra)
    return body


def row_sums(matrix: list[list[int]]) -> list[int]:
    return [sum(row) for row in matrix]


def col_sums(matrix: list[list[int]]) -> list[int]:
    return [sum(row[j] for row in matrix) for j in range(len(matrix[0]))]


def brute_force_min_change(
    counts: list[list[int]],
    base: int,
    row_totals: list[int],
    column_totals: list[int],
) -> int | None:
    """枚举所有“向上取整格”组合，返回满足目标合计的最小总改变量。"""
    rows = len(counts)
    cols = len(counts[0])
    lows = [[value - value % base for value in row] for row in counts]
    flexible = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] % base != 0
    ]
    best: int | None = None
    for size in range(len(flexible) + 1):
        for subset in combinations(flexible, size):
            rounded = [row[:] for row in lows]
            for i, j in subset:
                rounded[i][j] += base
            if row_sums(rounded) != row_totals:
                continue
            if col_sums(rounded) != column_totals:
                continue
            change = sum(
                abs(rounded[i][j] - counts[i][j])
                for i in range(rows)
                for j in range(cols)
            )
            if best is None or change < best:
                best = change
    return best


def assert_optimal_solution(body: dict, counts, base, row_totals, column_totals):
    """校验响应矩阵合法（相邻倍数、合计达标）且总改变量达到暴力枚举的最优值。"""
    rounded = body["rounded_counts"]
    assert len(rounded) == len(counts)
    for original_row, rounded_row in zip(counts, rounded):
        assert len(rounded_row) == len(original_row)
        for original, value in zip(original_row, rounded_row):
            assert value % base == 0
            low = original - original % base
            assert value in (low, low + base)
    assert body["row_totals"] == row_sums(rounded) == row_totals
    assert body["column_totals"] == col_sums(rounded) == column_totals
    expected = brute_force_min_change(counts, base, row_totals, column_totals)
    assert expected is not None
    assert body["total_change"] == expected


# ---------------------------------------------------------------- 基本功能

def test_basic_rounding_matches_targets():
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()

    # 唯一可行解：第 2 列两格都必须向上，其余向下。
    assert body["rounded_counts"] == [[10, 10], [0, 10]]
    assert body["row_totals"] == [20, 10]
    assert body["column_totals"] == [10, 20]
    # |12-10| + |7-10| + |3-0| + |9-10| = 9
    assert body["total_change"] == 9
    assert body["base"] == 10
    assert set(body) == {
        "rounded_counts",
        "row_totals",
        "column_totals",
        "total_change",
        "base",
    }


def test_exact_multiples_stay_unchanged():
    # 所有格都是 base 的整倍数：唯一可行解是原矩阵，改变量为 0。
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=[[10, 20], [30, 40]],
            row_totals=[30, 70],
            column_totals=[40, 60],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[10, 20], [30, 40]]
    assert body["total_change"] == 0


def test_prefers_cheaper_direction_per_cell():
    # 8 距 10 比距 0 近：目标 10 时向上（改变量 2）。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[8]], row_totals=[10], column_totals=[10]),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[10]]
    assert body["total_change"] == 2

    # 同一格目标为 0 时向下（改变量 8）。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[8]], row_totals=[0], column_totals=[0]),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[0]]
    assert body["total_change"] == 8


def test_base_one_keeps_original_matrix():
    # base=1 时任何整数都是整倍数，目标合计必须等于原始行列和。
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=[[3, 4], [5, 6]],
            base=1,
            row_totals=[7, 11],
            column_totals=[8, 10],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[3, 4], [5, 6]]
    assert body["total_change"] == 0


def test_zero_totals_round_everything_down():
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=[[3, 6], [8, 1]],
            row_totals=[0, 0],
            column_totals=[0, 0],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[0, 0], [0, 0]]
    assert body["total_change"] == 18


def test_supports_maximum_20_by_20_matrix():
    counts = [[10] * 20 for _ in range(20)]
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=counts,
            row_totals=[200] * 20,
            column_totals=[200] * 20,
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == counts
    assert body["total_change"] == 0


def test_supports_large_values():
    large = 10**12
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=[[large + 3]],
            row_totals=[large + 10],
            column_totals=[large + 10],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rounded_counts"] == [[large + 10]]
    assert body["total_change"] == 7


def test_optimal_against_brute_force():
    # 多组小矩阵与暴力枚举比对：响应必须可行且总改变量等于最优值。
    cases = [
        ([[12, 7], [3, 9]], 10, [20, 10], [10, 20]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 5, [10, 15, 20], [10, 15, 20]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 5, [5, 20, 20], [15, 15, 15]),
        ([[6, 11], [14, 9]], 5, [15, 25], [20, 20]),
        ([[2, 4, 6], [8, 10, 12]], 4, [12, 32], [8, 16, 20]),
        ([[7, 7, 7], [7, 7, 7], [7, 7, 7]], 10, [20, 20, 20], [20, 20, 20]),
        ([[13, 0, 17], [5, 22, 8]], 10, [30, 30], [20, 20, 20]),
    ]
    for counts, base, row_totals, column_totals in cases:
        response = client.post(
            ENDPOINT,
            json=payload(
                counts=counts,
                base=base,
                row_totals=row_totals,
                column_totals=column_totals,
            ),
        )
        assert response.status_code == 200, (counts, response.json())
        assert_optimal_solution(
            response.json(), counts, base, row_totals, column_totals
        )


def test_tie_cost_returns_some_optimal_solution():
    # 5 向上、向下改变量相同（均为 5）：两种目标各自唯一可行。
    for target in (0, 10):
        response = client.post(
            ENDPOINT,
            json=payload(
                counts=[[5]],
                row_totals=[target],
                column_totals=[target],
            ),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["rounded_counts"] == [[target]]
        assert body["total_change"] == 5


# ---------------------------------------------------------------- 422：counts 格式

def test_422_on_empty_counts():
    assert client.post(ENDPOINT, json=payload(counts=[])).status_code == 422


def test_422_on_counts_not_a_matrix():
    assert client.post(ENDPOINT, json=payload(counts="x")).status_code == 422
    assert client.post(ENDPOINT, json=payload(counts=5)).status_code == 422


def test_422_on_too_many_rows():
    counts = [[0] for _ in range(21)]
    response = client.post(
        ENDPOINT,
        json=payload(counts=counts, row_totals=[0] * 21, column_totals=[0]),
    )
    assert response.status_code == 422


def test_422_on_too_many_columns():
    counts = [[0] * 21]
    response = client.post(
        ENDPOINT,
        json=payload(counts=counts, row_totals=[0], column_totals=[0] * 21),
    )
    assert response.status_code == 422


def test_422_on_empty_row():
    assert client.post(ENDPOINT, json=payload(counts=[[]])).status_code == 422


def test_422_on_ragged_matrix():
    response = client.post(ENDPOINT, json=payload(counts=[[1, 2], [3]]))
    assert response.status_code == 422


def test_422_on_negative_count():
    response = client.post(ENDPOINT, json=payload(counts=[[1, -2], [3, 4]]))
    assert response.status_code == 422


def test_422_on_non_integer_count():
    for bad in (True, 1.5, "3", None):
        response = client.post(ENDPOINT, json=payload(counts=[[bad]]))
        assert response.status_code == 422, bad


# ---------------------------------------------------------------- 422：base 格式

def test_422_on_non_positive_base():
    for bad in (0, -5):
        response = client.post(ENDPOINT, json=payload(base=bad))
        assert response.status_code == 422, bad


def test_422_on_non_integer_base():
    for bad in (True, 2.5, "10"):
        response = client.post(ENDPOINT, json=payload(base=bad))
        assert response.status_code == 422, bad


# ---------------------------------------------------------------- 422：目标合计格式

def test_422_on_row_totals_length_mismatch():
    response = client.post(ENDPOINT, json=payload(row_totals=[20]))
    assert response.status_code == 422
    response = client.post(ENDPOINT, json=payload(row_totals=[10, 10, 10]))
    assert response.status_code == 422


def test_422_on_column_totals_length_mismatch():
    response = client.post(ENDPOINT, json=payload(column_totals=[10, 10, 10]))
    assert response.status_code == 422


def test_422_on_negative_total():
    response = client.post(ENDPOINT, json=payload(row_totals=[-10, 40]))
    assert response.status_code == 422


def test_422_on_total_not_multiple_of_base():
    response = client.post(ENDPOINT, json=payload(row_totals=[15, 15]))
    assert response.status_code == 422
    response = client.post(ENDPOINT, json=payload(column_totals=[10, 23]))
    assert response.status_code == 422


def test_422_on_non_integer_total():
    for bad in (True, 2.5, "20"):
        response = client.post(ENDPOINT, json=payload(row_totals=[bad, 30]))
        assert response.status_code == 422, bad


def test_422_on_unequal_side_sums():
    response = client.post(
        ENDPOINT, json=payload(row_totals=[20, 20], column_totals=[10, 20])
    )
    assert response.status_code == 422


def test_422_on_missing_or_extra_body_field():
    body = payload()
    for key in ("counts", "base", "row_totals", "column_totals"):
        incomplete = {k: v for k, v in body.items() if k != key}
        assert client.post(ENDPOINT, json=incomplete).status_code == 422, key
    assert client.post(ENDPOINT, json=payload(extra=1)).status_code == 422


# ---------------------------------------------------------------- 422：不可行

def test_422_when_row_total_below_all_down_rounding():
    # 第 2 行向下舍入和为 5（8 -> 5），目标 0 不可达。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[3, 8]], row_totals=[0], column_totals=[0]),
    )
    assert response.status_code == 422


def test_422_when_row_total_above_all_up_rounding():
    # 单行单格最多向上到 5，目标 10 不可达。
    response = client.post(
        ENDPOINT,
        json=payload(counts=[[3]], base=5, row_totals=[10], column_totals=[10]),
    )
    assert response.status_code == 422


def test_422_when_column_total_out_of_range():
    # 第 1 列向下舍入和为 10（12 -> 10、3 -> 0），目标 0 不可达。
    response = client.post(
        ENDPOINT,
        json=payload(row_totals=[10, 0], column_totals=[0, 10]),
    )
    assert response.status_code == 422


def test_422_when_row_and_column_constraints_conflict():
    # 单侧合计都在可取范围内，但行列约束无法同时满足：
    # 第 3 列只能由 (3,3) 向上提供 5，而第 3 行目标不允许任何格向上。
    response = client.post(
        ENDPOINT,
        json=payload(
            counts=[[1, 2, 5], [3, 4, 10], [0, 5, 6]],
            base=5,
            row_totals=[15, 20, 10],
            column_totals=[10, 10, 25],
        ),
    )
    assert response.status_code == 422


def test_422_detail_mentions_reason():
    response = client.post(
        ENDPOINT, json=payload(row_totals=[20, 20], column_totals=[10, 20])
    )
    assert response.status_code == 422
    assert "相等" in response.json()["detail"]


# ---------------------------------------------------------------- 既有服务不受影响

def test_existing_endpoints_still_available():
    assert client.get("/api/v1/health").status_code == 200
    root = client.get("/").json()
    assert root["round_count_table"] == "/api/v1/round-count-table"
