from typing import Tuple
from PIL import Image
import numpy as np
import cv2


def gen_binary_map(
    wh: Tuple[int, int],
    cxy: Tuple[float, float],
    r: float,
    data_type: np.dtype = np.float32,
):
    w, h = wh
    cx, cy = cxy
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=data_type)
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    distmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    bmap = np.zeros_like(distmap)
    bmap[distmap <= r**2] = 1
    return bmap.astype(data_type)


def gen_line_binary_map(
    wh: Tuple[int, int],
    cxy: Tuple[float, float],
    ang: float,  # in degrees
    l: float,
    r: float,
    data_type: np.dtype = np.float32,
):
    """
    Generate a binary map with a line centered at cxy, angled at ang, with half-length l,
    and width determined by r (radius around the line).
    """
    w, h = wh
    cx, cy = cxy

    # Return a blank map if the center is outside the image bounds
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return np.zeros((h, w), dtype=data_type)

    # Compute the endpoints of the line
    dx = l * np.cos(np.radians(ang))
    dy = l * np.sin(np.radians(ang))
    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy

    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    bmap = np.zeros((h, w), dtype=data_type)
    if l == 0:
        dist_to_p1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        bmap[dist_to_p1 <= 3] = 1
    else:
        for i in range(int(l)):
            dx = i * np.cos(np.radians(ang))
            dy = i * np.sin(np.radians(ang))
            x1, y1 = cx - dx, cy - dy
            x2, y2 = cx + dx, cy + dy
            # Distance from each pixel to the endpoints
            dist_to_p1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            dist_to_p2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            within_point1 = dist_to_p1 <= r
            within_point2 = dist_to_p2 <= r
            bmap[within_point1] = 1
            bmap[within_point2] = 1
    return bmap


def gen_heatmap(
    wh: Tuple[int, int],
    cxy: Tuple[float, float],
    r: float,
    data_type: np.dtype = np.float32,
    min_value: float = 0.7,
):
    w, h = wh
    cx, cy = cxy
    if cx < 0 or cy < 0:
        return np.zeros((h, w), dtype=data_type)
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    distmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    r2 = r**2
    heatmap = np.exp(-distmap / r2) / np.exp(-1.0) * min_value
    heatmap[heatmap < 0.5] = 0
    heatmap[heatmap > 1] = 1.0
    return heatmap.astype(data_type)


def gen_line_heatmap(
    wh: Tuple[int, int],
    cxy: Tuple[float, float],
    ang: float,  # in degrees
    l: float,  # half-length of the line
    r: float,  # radius around the line
    data_type: np.dtype = np.float32,
    min_value: float = 0.7,
):
    """
    Generate a heatmap for a line centered at cxy, angled at ang, with half-length l,
    and radius r (spread around the line).
    """
    w, h = wh
    cx, cy = cxy

    # Return a blank map if the center is outside the image bounds
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return np.zeros((h, w), dtype=data_type)

    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    heatmap = np.zeros((h, w), dtype=data_type)

    # Generate the line using incremental points
    for i in range(int(2 * l) + 1):  # Sample points along the line
        offset = i - l  # Offset from center (line is symmetric around center)
        dx = offset * np.cos(np.radians(ang))
        dy = offset * np.sin(np.radians(ang))
        px, py = cx + dx, cy + dy  # Current line point

        # Compute distance to the current point
        dist_to_point = np.sqrt((x - px) ** 2 + (y - py) ** 2)

        # Update heatmap values using exponential decay
        r2 = r**2
        contribution = np.exp(-(dist_to_point**2) / r2) / np.exp(-1.0) * min_value
        heatmap = np.maximum(
            heatmap, contribution
        )  # Take maximum contribution for overlapping regions

    # Clamp heatmap values
    heatmap[heatmap < 0.5] = 0
    heatmap[heatmap > 1] = 1.0

    return heatmap.astype(data_type)
