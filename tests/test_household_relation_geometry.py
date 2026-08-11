import torch

from cg_isaac_envs.tasks.household.relation_geometry import (
    cube_inside_mug_relative,
    mug_on_station_relative,
)


def test_cube_must_be_inside_cavity_not_above_or_outside_mug():
    positions = torch.tensor([
        [0.000, 0.000, 0.020],
        [0.000, 0.000, 0.100],
        [0.040, 0.000, 0.020],
        [0.000, 0.000, 0.000],
    ])
    assert cube_inside_mug_relative(positions).tolist() == [True, False, False, False]


def test_mug_must_be_on_station_not_hovering_or_outside():
    positions = torch.tensor([
        [0.000, 0.000, 0.003],
        [0.000, 0.000, 0.100],
        [0.080, 0.080, 0.003],
    ])
    assert mug_on_station_relative(positions, "cold_serving_place").tolist() == [True, False, False]


def test_station_checks_match_circle_square_and_rectangle_footprints():
    hot = torch.tensor([[0.070, 0.000, 0.003], [0.070, 0.070, 0.003]])
    storage = torch.tensor([[0.085, 0.045, 0.003], [0.000, 0.060, 0.003]])
    assert mug_on_station_relative(hot, "hot_serving_place").tolist() == [True, False]
    assert mug_on_station_relative(storage, "storage_place").tolist() == [True, False]
