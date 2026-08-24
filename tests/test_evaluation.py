import torch

from token_cx.evaluation import auc


def test_auc_uses_unit_interval():
    assert auc([0.0, 0.5, 1.0]) == 0.5


def test_auc_rejects_single_value():
    try:
        auc(torch.tensor([1.0]))
    except ValueError:
        return

    raise AssertionError("AUC must reject a single point")
