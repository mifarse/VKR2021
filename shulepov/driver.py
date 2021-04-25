from multiprocessing import Pool
from .main import *
from enum import IntEnum
import hashlib
import pickle


class CalcMethod(IntEnum):
    DELAUNAY = 0  # Метод Делоне.
    SUBSEQUENT = 1  # Метод последовательной триангуляции.


class Driver:
    HOME_DIR = pathlib.Path("/home/urukov")  # Home directory.
    CACHE_DIR = HOME_DIR / "utils" / ".utils_cache"

    def __init__(self):
        self._method: CalcMethod = None
        self._import_dir: pathlib.Path = None
        self._export_dir: pathlib.Path = None
        self._data_grid = self.setup_data_grid()
        self.pca_results = []

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
            with open(self.CACHE_DIR / hashsum, "rb") as f:
                pca_result = pickle.load(f)
        else:
            df = pd.read_pickle(fname["path"])
            df = self._remove_outliers(df)
            pca_result = PCA(n_components=2).fit_transform(df)
            with open(self.CACHE_DIR / hashsum, "wb") as f:
                pickle.dump(pca_result, f)

        finish = time.monotonic() - start
        print(f'🏁 [{finish:.2f}s] Готово! Из кэша = {from_cache}, {fname["name"]}')
        return {fname["name"]: pca_result}
        # return None

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
            tmp[k] = v
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