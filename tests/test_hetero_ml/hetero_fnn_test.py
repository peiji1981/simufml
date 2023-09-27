import pytest
from simufml.utils.util import RANDOM_SEED
from simufml.tests.util import approx_eq
from simufml.demo import hetero_fnn_demo

lr = 1e-2
wd = 1e-2
train_bs = 16
valid_bs = 64
rand_seed = 7

RANDOM_SEED.seed = rand_seed

@pytest.mark.parametrize(
    ('skip_cipher', 'epochs', 'expect_loss', 'expect_metric'), 
    [(True, 10, 0.5406, 0.9837)]
)
def test_hetero_fnn(skip_cipher, epochs, expect_loss, expect_metric):
    loss, metric = hetero_fnn_demo(
        skip_cipher, epochs, train_bs, valid_bs, rand_seed, lr, wd, False
    )

    assert approx_eq(loss, expect_loss, expect_loss*1e-2)
    assert approx_eq(metric, expect_metric, expect_metric*1e-2)
