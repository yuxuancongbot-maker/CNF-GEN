import torch

from cnf_motion.video_wm import VideoWorldModel6D, foreground_weight


def test_forward_shapes():
    model = VideoWorldModel6D(hidden=8)
    first = torch.rand(2, 3, 32, 32)
    control = torch.rand(2, 3, 5, 32, 32)
    q = torch.randn(2, 5, 7)
    output = model(first, control, q)
    assert output.shape == (2, 5, 3, 32, 32)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_foreground_weight():
    control = torch.zeros(1, 3, 4, 8, 8)
    control[:, :, :, 2:4, 2:4] = 1.0
    weight = foreground_weight(control)
    assert weight.shape == (1, 4, 8, 8)
    assert float(weight.max()) == 11.0
    assert float(weight.min()) == 1.0
