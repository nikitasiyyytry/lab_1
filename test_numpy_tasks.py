import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    plot_histograms,
    plot_shapes,
    sum_prod,
    unique_columns,
    unique_rows,
)


class NumpyTasksTests(unittest.TestCase):
    def test_sum_prod(self):
        matrices = np.array([np.eye(2), [[2, 0], [0, 3]]])
        vectors = np.array([[[1], [2]], [[3], [4]]])
        np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(matrices, vectors)), [[7], [14]])

    def test_binarize_returns_new_matrix(self):
        source = np.array([[0.5, 0.6], [-1, 2]])
        result = binarize(BinarizeInput(source, 0.5))
        np.testing.assert_array_equal(result, [[0, 1], [0, 1]])
        self.assertEqual(source[0, 1], 0.6)

    def test_unique_rows_and_columns(self):
        matrix = np.array([[2, 1, 2], [2, 3, 3]])
        self.assertEqual(unique_rows(MatrixInput(matrix)), [[1, 2], [2, 3]])
        self.assertEqual(unique_columns(MatrixInput(matrix)), [[2], [1, 3], [2, 3]])

    def test_matrix_statistics(self):
        result = matrix_statistics(RandomMatrixInput(2, 3, mean=5, std=0, seed=9))
        np.testing.assert_array_equal(result.matrix, np.full((2, 3), 5.0))
        np.testing.assert_array_equal(result.row_means, [5, 5])
        np.testing.assert_array_equal(result.column_variances, [0, 0, 0])

    def test_histograms_are_built_for_every_row_and_column(self):
        pyplot = ModuleType("matplotlib.pyplot")
        pyplot.figure = MagicMock(side_effect=[object() for _ in range(5)])
        pyplot.hist = MagicMock()
        pyplot.title = MagicMock()
        pyplot.xlabel = MagicMock()
        pyplot.ylabel = MagicMock()
        pyplot.show = MagicMock()
        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = pyplot

        with patch.dict("sys.modules", {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
            figures = plot_histograms(np.array([[1, 2, 3], [4, 5, 6]]), show=False)

        self.assertEqual(len(figures), 5)
        self.assertEqual(pyplot.hist.call_count, 5)
        pyplot.show.assert_not_called()

    def test_chess(self):
        expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        np.testing.assert_array_equal(chess(ChessInput(3, 4, 0, 1)), expected)

    def test_rectangle(self):
        result = draw_rectangle(RectangleInput(2, 3, 5, 6, (255, 0, 0), (0, 0, 0)))
        self.assertEqual(result.shape, (5, 6, 3))
        self.assertEqual(np.count_nonzero(np.all(result == (255, 0, 0), axis=2)), 6)

    def test_ellipse(self):
        result = draw_ellipse(EllipseInput(1, 1, 5, 5, (1, 2, 3), (0, 0, 0)))
        expected_mask = np.zeros((5, 5), dtype=bool)
        expected_mask[2, 1:4] = True
        expected_mask[1:4, 2] = True
        np.testing.assert_array_equal(np.all(result == (1, 2, 3), axis=2), expected_mask)

    def test_rectangle_and_ellipse_visualization(self):
        figure = MagicMock()
        axes = [MagicMock(), MagicMock()]
        pyplot = ModuleType("matplotlib.pyplot")
        pyplot.subplots = MagicMock(return_value=(figure, axes))
        pyplot.show = MagicMock()
        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        rectangle_data = RectangleInput(2, 2, 6, 6, (255, 0, 0), (0, 0, 0))
        ellipse_data = EllipseInput(2, 1, 6, 6, (0, 255, 0), (0, 0, 0))

        with patch.dict("sys.modules", {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
            result = plot_shapes(rectangle_data, ellipse_data, show=False)

        self.assertIs(result, figure)
        axes[0].imshow.assert_called_once()
        axes[1].imshow.assert_called_once()
        axes[0].axis.assert_called_once_with("off")
        axes[1].axis.assert_called_once_with("off")
        pyplot.show.assert_not_called()

    @unittest.skipIf(plt is None, "Matplotlib is not installed")
    def test_visual_output_for_tasks_4_and_6(self):
        """Show the histograms and shapes; close the windows to finish the tests."""
        statistics = matrix_statistics(
            RandomMatrixInput(2, 3, mean=0, std=1, seed=42)
        )
        plot_histograms(statistics.matrix, show=False)

        rectangle_data = RectangleInput(
            120, 70, 200, 300, (70, 110, 230), (235, 235, 245)
        )
        ellipse_data = EllipseInput(
            70, 45, 200, 300, (50, 160, 110), (235, 235, 245)
        )
        plot_shapes(rectangle_data, ellipse_data, show=False)

        plt.show()

    def test_time_series(self):
        result = analyze_time_series(TimeSeriesInput(np.array([1, 3, 2, 4, 0]), 3))
        np.testing.assert_array_equal(result.local_maxima_indices, [1, 3])
        np.testing.assert_array_equal(result.local_minima_indices, [2])
        np.testing.assert_allclose(result.moving_average, [2, 3, 2])

    def test_one_hot(self):
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        np.testing.assert_array_equal(one_hot(OneHotInput(np.array([0, 2, 3, 0]))), expected)
        self.assertEqual(one_hot(OneHotInput(np.array([], dtype=int))).shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
