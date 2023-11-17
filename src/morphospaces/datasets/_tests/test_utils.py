import numpy as np

from morphospaces.datasets.utils import FilterSliceBuilder, SliceBuilder


def make_test_array() -> np.ndarray:
    array = np.zeros((20, 20, 20))
    array[0:10, 0:10, 0:10] = 1
    array[10:20, 0:10, 0:10] = 2
    array[10:20, 10:12, 10:12] = 1

    return array


def test_slice_builder():
    """Test SliceBuilder without filtering"""
    patch_shape = (10, 10, 10)
    stride = (10, 10, 10)

    expected_patches = [
        (slice(0, 10, None), slice(0, 10, None), slice(0, 10, None)),
        (slice(10, 20, None), slice(0, 10, None), slice(0, 10, None)),
        (slice(0, 10, None), slice(10, 20, None), slice(0, 10, None)),
        (slice(0, 10, None), slice(0, 10, None), slice(10, 20, None)),
        (slice(10, 20, None), slice(10, 20, None), slice(0, 10, None)),
        (slice(0, 10, None), slice(10, 20, None), slice(10, 20, None)),
        (slice(10, 20, None), slice(0, 10, None), slice(10, 20, None)),
        (slice(10, 20, None), slice(10, 20, None), slice(10, 20, None)),
    ]

    array = make_test_array()
    slice_builder = SliceBuilder(
        dataset=array, patch_shape=patch_shape, stride_shape=stride
    )

    assert len(slice_builder.slices) == 8

    for patch in expected_patches:
        assert patch in slice_builder.slices


def test_filter_slice_builder_no_ignore_index():
    """Test that the FilterSliceBuilder uses the threshold correctly."""
    patch_shape = (10, 10, 10)
    stride = (10, 10, 10)
    threshold = 0.5

    # set slack acceptance such that no patches that fail threshold will be
    # randomly selected
    slack_acceptance = 0

    expected_patches = [
        (slice(0, 10, None), slice(0, 10, None), slice(0, 10, None)),
        (slice(10, 20, None), slice(0, 10, None), slice(0, 10, None)),
    ]

    array = make_test_array()

    slice_builder = FilterSliceBuilder(
        dataset=array,
        patch_shape=patch_shape,
        stride_shape=stride,
        threshold=threshold,
        slack_acceptance=slack_acceptance,
    )

    assert len(slice_builder.slices) == 2

    for patch in expected_patches:
        assert patch in slice_builder.slices


def test_filter_slice_builder_ignore_index():
    """Test that the FilterSliceBuilder properly ignores indices."""
    patch_shape = (10, 10, 10)
    stride = (10, 10, 10)
    threshold = 0.5
    ignore_index = (0, 2)

    # set slack acceptance such that no patches that fail threshold will be
    # randomly selected
    slack_acceptance = 0

    expected_patches = [
        (slice(0, 10, None), slice(0, 10, None), slice(0, 10, None)),
    ]

    array = make_test_array()

    slice_builder = FilterSliceBuilder(
        dataset=array,
        patch_shape=patch_shape,
        stride_shape=stride,
        threshold=threshold,
        filter_ignore_index=ignore_index,
        slack_acceptance=slack_acceptance,
    )

    assert len(slice_builder.slices) == 1

    for patch in expected_patches:
        assert patch in slice_builder.slices
