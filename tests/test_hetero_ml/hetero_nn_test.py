import pytest
from simufml.utils.util import RANDOM_SEED
from simufml.tests.util import approx_eq
from simufml.demo import hetero_nn_demo

lr = 1e-2
wd = 1e-2
train_bs = 16
valid_bs = 64
rand_seed = 7

RANDOM_SEED.seed = 7

@pytest.mark.parametrize(
    ('skip_cipher', 'epochs', 'expect_loss', 'expect_metric'), 
    [
        (True, 10, 0.6477, 0.9602), 
        (False, 1, 0.6784, 0.1009)
    ]
)
def test_hetero_nn(skip_cipher, epochs, expect_loss, expect_metric):
    loss, metric = hetero_nn_demo(
        skip_cipher, epochs, train_bs, valid_bs, rand_seed, lr, wd, False
    )

    assert approx_eq(loss, expect_loss, expect_loss*1e-2)
    assert approx_eq(metric, expect_metric, expect_metric*1e-2)
