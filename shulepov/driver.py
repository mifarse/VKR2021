from .main import *
from enum import IntEnum


class CalcMethod(IntEnum):
    DELAUNAY = 0  # Метод Делоне.
    SUBSEQUENT = 1  # Метод последовательной триангуляции.


class Driver:
    def __init__(self):
        self._method: CalcMethod = None
        self._file_names: list = []
        self._data_grid = self.setup_data_grid()

        print("🔌 Shulepov driver connected!")

    def main(self):
        """
        Calls main() function from this file.

        """
        assert self._method is not None, "🚫 Метод не указан!"
        print(f"⛏️ Выбран метод {self._method}")
        print("📁 Выбраны файлы")
        for file_name in self._file_names:
            print(f"\t📜 {file_name}")
        main(
            {
                "method": self._method,
                "file_names": self._file_names,
                "data_grid": self._data_grid,
            }
        )

    @property
    def file_names(self):
        return self._file_names

    @file_names.setter
    def file_names(self, value):
        self._file_names = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value: CalcMethod):
        self._method = value

    def setup_data_grid(self):
        return [
            ["All", [(1, 13)], [(0, 7200)]],
        ]