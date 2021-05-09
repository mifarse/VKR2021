from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from .main import *
from enum import IntEnum
import hashlib
import pickle
import time
from scipy.spatial import Delaunay

# graphics
import matplotlib.pyplot as plt


def triangle_area(points):
    """Calculates area of triangle by 3 points.

    Args:
        points (list): List of 2d points: [point1, point2, point3]

    Returns:
        float: Area of triangle.
    """

    x1 = points[0][0]
    x2 = points[1][0]
    x3 = points[2][0]
    y1 = points[0][1]
    y2 = points[1][1]
    y3 = points[2][1]
    a = np.array([[x1 - x3, y1 - y3], [x2 - x3, y2 - y3]])
    return np.abs(0.5 * np.linalg.det(a))


class CalcMethod(IntEnum):
    DELAUNAY = 0  # Метод Делоне.
    SUBSEQUENT = 1  # Метод последовательной триангуляции.


class Graphy:
    """It used to visualise PCA data."""

    @staticmethod
    def scatter_areas(areas):
        return plt.scatter(areas.index, areas)


class PCAMetric:
    """Custom class encapsulates dataframe with pca and adds some features."""

    def __init__(self):
        pass

    def __init__(self, df):
        self.df = df

    def by_weeks(self):
        """Returns list of dataframes, grouped by week.

        Returns:
            list: each is pd.Dataframe.
        """
        return [PCAMetric(g) for n, g in self.df.groupby(pd.Grouper(freq="W"))]

    def by_months(self):
        return [PCAMetric(g) for n, g in self.df.groupby(pd.Grouper(freq="M"))]

    def total_delaunay_area(self):
        """Splits data into delaunay triangles. Calculates area of each, return total
        area of the resulting figure."""
        # Делоне объект
        tri = Delaunay(self.df)
        # tri.simplices - это такой массив, который содержит 3 индекса, исходного масси-
        # ва, которые формируют треугольник.
        areas_list = []

        # Точки треугольников. Каждый треугольник - ((х1, у1), (х2, у2), (х3, у3))
        trinangle_points = [
            (
                (self.df.iloc[simplice[0]][0], self.df.iloc[simplice[0]][1]),
                (self.df.iloc[simplice[1]][0], self.df.iloc[simplice[1]][1]),
                (self.df.iloc[simplice[2]][0], self.df.iloc[simplice[2]][1]),
            )
            for simplice in tri.simplices
        ]

        # Треды нифига не ускоряют, ну ладно
        with ThreadPool(4) as pool:
            areas_list = pool.map(triangle_area, trinangle_points)

        # Площадь всех треугольников в одной серии
        return sum(areas_list)


class Driver:
    HOME_DIR = pathlib.Path("/home/urukov")  # Home directory.
    CACHE_DIR = HOME_DIR / ".utils_cache"

    def __init__(self):
        self._method: CalcMethod = None
        self._import_dir: pathlib.Path = None
        self._export_dir: pathlib.Path = None
        self._data_grid = self.setup_data_grid()
        # Содержит массив PCAMetric.
        self.pca_results: list = []

        # сюда накидаем кэш. Все вычисления будем сохранять, чтоб быстрее перезапускать
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        print("🔌 Shulepov driver connected!")

    def main(self):
        """
        Calls main() function from this file.

        """
        assert self._method is not None, "🚫 Метод не указан!"
        print(f"⛏️ Выбран метод {self._method}")
        main(
            {
                "method": self._method,
                "data_grid": self._data_grid,
                "import_dir": self._import_dir,
                "export_dir": self._export_dir,
            }
        )

    @staticmethod
    def _remove_outliers(df):
        """Remove outliers from dataset. It can be commented if need."""
        return df.loc[((df > 5) & (df < 35)).all(axis=1)]

    def pca_job(self, fname):
        """Загружает 1 датафрейм и вычисляет PCA."""
        start = time.monotonic()

        # Эта переменная станет True, если найдется кэш
        from_cache = False
        # Сперва возьмем хеш файла и тут же откроем датасет
        with open(fname["path"], "rb") as f:
            file_bytes = f.read()  # read file as bytes
            hashsum = hashlib.md5(file_bytes).hexdigest()

        if os.path.exists(self.CACHE_DIR / hashsum):
            from_cache = True
            pca_df = pd.read_pickle(self.CACHE_DIR / hashsum)
        else:
            df = pd.read_pickle(fname["path"])

            # Здесь убираются крайние значения. Можно закомментировать, если надо.
            df = self._remove_outliers(df)

            pca_result = PCA(n_components=2).fit_transform(df)
            pca_df = pd.DataFrame(pca_result, index=df.index.copy())

            pca_df.to_pickle(self.CACHE_DIR / hashsum)

        finish = time.monotonic() - start
        print(f'🏁 [{finish:.2f}s] Готово! Из кэша = {from_cache}, {fname["name"]}')
        return {fname["name"]: pca_df}

    def prepare_pca(self):
        """Подготовим двумерные данные, полученные по PCA.
        Загоним их в датафреймы и закэшируем."""

        # Зачитаем файлы
        fnames = [
            {"path": f.path, "name": f.name} for f in os.scandir(self._import_dir)
        ]
        start = time.monotonic()

        # self.pca_job(fnames[0])
        with Pool(processes=4) as pool:
            self.pca_results = pool.map(self.pca_job, fnames)

        # Конвертим [{zone:pca}, {zone:pca}] в {zone:pca, zone:pca}
        tmp = dict()
        for i in self.pca_results:
            k, v = [(k, v) for k, v in i.items()][0]
            tmp[k] = PCAMetric(v)
        self.pca_results = tmp

        finish = time.monotonic() - start
        print(
            f"🏁🏁🏁 [{finish:.2f}s] PCA рассчитано для {len(self.pca_results)} датафреймов."
        )

    @property
    def import_dir(self):
        return self._import_dir

    @import_dir.setter
    def import_dir(self, value: pathlib.Path):
        self._import_dir = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value: CalcMethod):
        self._method = value

    @property
    def export_dir(self):
        return self._export_dir

    @export_dir.setter
    def export_dir(self, value: pathlib.Path):
        self._export_dir = value

    def setup_data_grid(self):
        return [
            ["All", [(1, 13)], [(0, 7200)]],
        ]