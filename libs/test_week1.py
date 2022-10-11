import libs.week1 as week1
import numpy as np
import pytest


def test_week1():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    row_strategy = np.array([[0.1, 0.2, 0.7]])
    column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

    row_value = week1.evaluate(matrix=matrix, row_strategy=row_strategy, column_strategy=column_strategy)
    assert row_value == pytest.approx(0.08)

    br_value_row = week1.best_response_value_row(matrix=matrix, row_strategy=row_strategy)
    br_value_column = week1.best_response_value_column(matrix=matrix, column_strategy=column_strategy)
    assert br_value_row == pytest.approx(-0.6)
    assert br_value_column == pytest.approx(-0.2)


