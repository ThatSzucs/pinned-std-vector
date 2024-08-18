import numpy as np
import pyplayground


def test_pageable_i8_vector():
    # Arrange
    count = 512

    # Act
    my_vec = pyplayground.PageableI8Vector(count)
    my_array = my_vec.as_ndarray()

    # Assert
    assert type(my_vec) == pyplayground.PageableI8Vector
    assert type(my_array) == np.ndarray
    assert my_array.dtype == np.int8
    assert my_array.shape == (count,)
    assert my_array.max() == min(count, np.iinfo(my_array.dtype).max)
    assert (
        my_array.min() == 0
        if count < np.iinfo(my_array.dtype).max
        else np.iinfo(my_array.dtype).min
    )


def test_pinned_i8_vector():
    # Arrange
    count = 512

    # Act
    my_vec = pyplayground.PinnedI8Vector(count)
    my_array = my_vec.as_ndarray()

    # Assert
    assert type(my_vec) == pyplayground.PinnedI8Vector
    assert type(my_array) == np.ndarray
    assert my_array.dtype == np.int8
    assert my_array.shape == (count,)
    assert my_array.max() == min(count, np.iinfo(my_array.dtype).max)
    assert (
        my_array.min() == 0
        if count < np.iinfo(my_array.dtype).max
        else np.iinfo(my_array.dtype).min
    )


# This code is part of the playgrounds project
# Copyright (c) 2024 ThatSzucs
# Distributed under the MIT license. See accompanying license file
# copy at https://opensource.org/licenses/MIT.
