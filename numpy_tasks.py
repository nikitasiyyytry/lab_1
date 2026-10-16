"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(column).tolist() for column in np.asarray(matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def plot_histograms(matrix: np.ndarray, show: bool = True) -> list[object]:
    """Build a separate value histogram for every row and column."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")

    figures = []
    for index, row in enumerate(matrix):
        figure = plt.figure()
        plt.hist(row, bins="auto", edgecolor="black")
        plt.title(f"Row {index + 1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        figures.append(figure)

    for index, column in enumerate(matrix.T):
        figure = plt.figure()
        plt.hist(column, bins="auto", edgecolor="black")
        plt.title(f"Column {index + 1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        figures.append(figure)

    if show:
        plt.show()
    return figures


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if width > image_width or height > image_height:
        raise ValueError("Rectangle must fit inside the image")

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = background_color
    start_x = (image_width - width) // 2
    start_y = (image_height - height) // 2
    image[start_y:start_y + height, start_x:start_x + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        raise ValueError("Ellipse semi-axes must be positive")

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    center_x = image_width // 2
    center_y = image_height // 2
    mask = (
        ((x - center_x) / semi_axis_x) ** 2
        + ((y - center_y) / semi_axis_y) ** 2
        <= 1
    )
    image[mask] = shape_color
    return image


def plot_shapes(
    rectangle_data: RectangleInput,
    ellipse_data: EllipseInput,
    show: bool = True,
) -> object:
    """Display the rectangle and ellipse images side by side."""
    import matplotlib.pyplot as plt

    rectangle_image = draw_rectangle(rectangle_data)
    ellipse_image = draw_ellipse(ellipse_data)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    for axis, image, title in zip(
        axes,
        (rectangle_image, ellipse_image),
        ("Rectangle", "Ellipse"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    if show:
        plt.show()
    return figure


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Time series must be one-dimensional")
    if not 1 <= window <= values.size:
        raise ValueError("Window must be between 1 and the series length")

    local_maxima = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    ) + 1
    local_minima = np.flatnonzero(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    ) + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")

    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integers")
    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative")

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if class_count < 0:
        raise ValueError("Class count must be non-negative")
    if labels.size and np.any(labels >= class_count):
        raise ValueError("A label is outside the requested class range")

    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
