"""Visual demonstration for NumPy task 6."""

from grader_contracts.numpy_tasks import EllipseInput, RectangleInput
from numpy_tasks import plot_shapes


rectangle = RectangleInput(
    width=120,
    height=70,
    image_height=200,
    image_width=300,
    shape_color=(70, 110, 230),
    background_color=(235, 235, 245),
)

ellipse = EllipseInput(
    semi_axis_x=70,
    semi_axis_y=45,
    image_height=200,
    image_width=300,
    shape_color=(50, 160, 110),
    background_color=(235, 235, 245),
)

plot_shapes(rectangle, ellipse)
