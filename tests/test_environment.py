import numpy as np
import scipy
import matplotlib


def test_scientific_environment():
    assert np.__version__
    assert scipy.__version__
    assert matplotlib.__version__


def test_basic_linear_algebra():
    matrix = np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ]
    )

    vector = np.array([1.0, 2.0])

    result = matrix @ vector

    np.testing.assert_allclose(result, [4.0, 7.0])
