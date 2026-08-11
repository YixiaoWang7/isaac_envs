"""Pure geometric predicates for household final-state success."""

from __future__ import annotations

import torch


MUG_INNER_RADIUS_M = 0.042
CUBE_HALF_SIZE_M = 0.010
MUG_BOTTOM_HEIGHT_M = 0.006
MUG_HEIGHT_M = 0.080
STATION_HORIZONTAL_TOLERANCE_M = 0.005
STATION_MAX_MUG_ROOT_HEIGHT_M = 0.035
STATION_MIN_MUG_ROOT_HEIGHT_M = -0.010
STATION_HALF_EXTENTS_M = {
    "hot_serving_place": (0.076, 0.076),
    "cold_serving_place": (0.067, 0.067),
    "storage_place": (0.090, 0.050),
}


def cube_inside_mug_relative(relative_position: torch.Tensor) -> torch.Tensor:
    """Check that a 20 mm cube center lies within the hollow mug cavity."""
    radial_limit = MUG_INNER_RADIUS_M - CUBE_HALF_SIZE_M + 0.003
    minimum_z = MUG_BOTTOM_HEIGHT_M + CUBE_HALF_SIZE_M - 0.005
    maximum_z = MUG_HEIGHT_M
    radial_distance = torch.linalg.vector_norm(relative_position[:, :2], dim=1)
    return (
        (radial_distance <= radial_limit)
        & (relative_position[:, 2] >= minimum_z)
        & (relative_position[:, 2] <= maximum_z)
    )


def mug_on_station_relative(relative_position: torch.Tensor, station_name: str) -> torch.Tensor:
    """Check mug-root contact height and center position against a station shape."""
    half_x, half_y = STATION_HALF_EXTENTS_M[station_name]
    valid_height = (
        (relative_position[:, 2] >= STATION_MIN_MUG_ROOT_HEIGHT_M)
        & (relative_position[:, 2] <= STATION_MAX_MUG_ROOT_HEIGHT_M)
    )
    if station_name == "hot_serving_place":
        horizontal = (
            torch.linalg.vector_norm(relative_position[:, :2], dim=1)
            <= half_x + STATION_HORIZONTAL_TOLERANCE_M
        )
    else:
        horizontal = (
            (relative_position[:, 0].abs() <= half_x + STATION_HORIZONTAL_TOLERANCE_M)
            & (relative_position[:, 1].abs() <= half_y + STATION_HORIZONTAL_TOLERANCE_M)
        )
    return horizontal & valid_height
