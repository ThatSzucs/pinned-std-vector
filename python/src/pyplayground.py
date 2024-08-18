import numpy as np
import playground_bindings as cpp


class PageableI8Vector:
    def __init__(self, count: int) -> None:
        self._cpp = cpp.PageableI8Vector(count)
        return

    def as_ndarray(self) -> np.ndarray:
        array = self._cpp.as_ndarray()
        return array


class PinnedI8Vector:
    def __init__(self, count: int) -> None:
        self._cpp = cpp.PinnedI8Vector(count)
        return

    def as_ndarray(self) -> np.ndarray:
        array = self._cpp.as_ndarray()
        return array
