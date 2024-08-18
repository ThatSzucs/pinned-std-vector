#include "pyarray_utils.h"

#include "cuda_runtime.h"
#include "pybind11/pybind11.h"

#include <cstdint>
#include <sstream>

namespace py = pybind11;

namespace playground {

template <typename T>
void init_pageable_vector(py::module &m, const std::string py_class_name)
{
    py::class_<std::vector<T>>(m, py_class_name.c_str())
        .def(py::init([](int count) {
            std::unique_ptr<std::vector<T>> vec = std::make_unique<std::vector<T>>();
            vec->reserve(count);
            for (int i = 0; i < count; ++i)
                vec->push_back(static_cast<T>(i)); // Do not care about overflow
            return vec;
        }))
        .def("as_ndarray", [](std::vector<T> &my_vector) { return as_numpy(my_vector); });
};

template <typename T>
void init_pinned_vector(py::module &m, const std::string py_class_name)
{
    py::class_<pinned_vector<T>>(m, py_class_name.c_str())
        .def(py::init([](int count) {
            std::unique_ptr<pinned_vector<T>> vec = std::make_unique<pinned_vector<T>>();
            vec->reserve(count);
            for (int i = 0; i < count; ++i)
                vec->push_back(static_cast<T>(i)); // Do not care about overflow
            return vec;
        }))
        .def("as_ndarray", [](pinned_vector<T> &my_vector) { return as_numpy(my_vector); });
};

PYBIND11_MODULE(playground_bindings, m)
{
    init_pageable_vector<int8_t>(m, "PageableI8Vector");
    init_pinned_vector<int8_t>(m, "PinnedI8Vector");
}

} // namespace playground

/*
 *  This code is part of the playground project.
 *  Copyright (c) 2024 ThatSzucs
 *
 *  Distributed under the MIT license. See accompanying license file
 *  or copy at https://opensource.org/licenses/MIT.
 */
