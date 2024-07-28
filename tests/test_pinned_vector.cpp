#include "playground/pinned_vector.h"

#include <Catch2/catch.hpp>

#include <vector>

namespace playground {

using TileTypes = std::tuple<float, double, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

TEMPLATE_LIST_TEST_CASE("Init (with values)", "[pinned_vector]", TileTypes)
{
    // Arrange
    constexpr int expected_size = 3;
    auto expected_vector = std::vector<TestType>{1, 2, 3};

    // Act
    auto test_vector = pinned_vector<TestType>{1, 2, 3};

    // Assert
    REQUIRE(test_vector.size() == expected_size);
    for (int i = 0; i < test_vector.size(); ++i)
        REQUIRE(test_vector[i] == expected_vector[i]);
}

TEMPLATE_LIST_TEST_CASE("Init (with size and values)", "[pinned_vector]", TileTypes)
{
    // Arrange
    constexpr int expected_size = 3;
    constexpr int expected_value = 8;

    // Act
    auto test_vector = pinned_vector<TestType>(expected_size, expected_value);

    // Assert
    REQUIRE(test_vector.size() == expected_size);
    for (int i = 0; i < test_vector.size(); ++i)
        REQUIRE(test_vector[i] == expected_value);
}

TEMPLATE_LIST_TEST_CASE("push_back", "[pinned_vector]", TileTypes)
{
    // Arrange
    constexpr int expected_size = 17;
    auto test_vector = pinned_vector<TestType>();

    // Act
    for (int i = 0; i < expected_size; ++i)
        test_vector.push_back(i);

    // Assert
    REQUIRE(test_vector.size() == expected_size);
    REQUIRE(test_vector.capacity() >= expected_size);
    for (int i = 0; i < test_vector.size(); ++i)
        REQUIRE(test_vector[i] == i);
}

} // namespace playground

/*
 *  This code is part of the playground project.
 *  Copyright (c) 2024 ThatSzucs
 *
 *  Distributed under the MIT license. See accompanying license file
 *  or copy at https://opensource.org/licenses/MIT.
 */
