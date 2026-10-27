"""Common chart input checks before native renderer dispatch."""
import math


def validate(data, config):
    paired = {'bar', 'horizontal_bar', 'pie', 'donut', 'line', 'area', 'scatter', 'funnel'}
    if config.chart_type.value in paired and data.labels and len(data.labels) != len(data.values):
        raise ValueError('Chart labels and values must have matching lengths')
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in data.values):
        raise ValueError('Chart values must be finite numbers')
    if any(not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0 for size in (config.width, config.height)):
        raise ValueError('Chart dimensions must be positive')
    if config.chart_type.value in {'pie', 'donut'} and any(value < 0 for value in data.values):
        raise ValueError('Pie chart values must be non-negative')
