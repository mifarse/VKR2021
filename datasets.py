import os
import pandas as pd
import time

from .building import Building
from .constants import *


def funny_apts_str(units):
    s = ""
    for unit in units:
        s += f"🚪 {unit['unit_number']}, "
    return s[:-2] + "."


class Datasets:
    def __init__(self, building: Building):
        self.building = building

    def _merge_temps(self, unit_refs, time_from: str, time_to: str):

        # Обрабатываем первый датасет и к нему начинаем прилеплять другие
        first_unit = unit_refs[0]
        # Ограничиваем по временным промежуткам, потому что биг дата все-таки
        time_from = pd.Timestamp(time_from, tz="America/New_York")
        time_to = pd.Timestamp(time_to, tz="America/New_York")
        time_filter = lambda index: (time_from <= index) & (index <= time_to)

        # Открываем файл, устанавливаем ключевое поле, мерджим средним дубликаты, обрезаем по времени.
        filepath = BIN_DIR / self.building.name / "%s.csv.pkl.gz"
        column_name = "unit_%s"
        ds = pd.read_pickle(str(filepath) % first_unit["name_opc"])
        ds.rename(
            columns={"value": column_name % first_unit["hmi_unit_number"]}, inplace=True
        )
        ds.set_index("time", inplace=True)
        ds = ds[time_filter(ds.index)]
        ds = ds.groupby(ds.index).mean()

        for i, ur in enumerate(unit_refs[1:]):
            # Открываем файл, устанавливаем ключевое поле, мерджим средним дубликаты, обрезаем по времени.
            try:
                iter_ds = pd.read_pickle(str(filepath) % ur["name_opc"])
            except FileNotFoundError:
                print(f"🙄 Файл не найден - {ur['name_opc']}.")
                continue
            iter_ds.rename(
                columns={"value": column_name % ur["hmi_unit_number"]}, inplace=True
            )
            iter_ds.set_index("time", inplace=True)
            iter_ds = iter_ds[time_filter(iter_ds.index)]
            iter_ds = iter_ds.groupby(iter_ds.index).mean()

            # Склеиваем с основным датасетом.
            ds = pd.merge(ds, iter_ds, how="outer", left_index=True, right_index=True)
            progress = (i + 1) * 100 / len(unit_refs[1:])
            print(f"Progress: {progress:.2f}% ", end="\x1b[1K\r")

        # У датасета остаются пустые поля от OUTER объединения, заполним их.
        ds.fillna(method="ffill", inplace=True)
        ds.fillna(method="bfill", inplace=True)
        return ds

    def create_temperature_datasets_by_zones(self, time_from: str, time_to: str):
        """Создает несколько датасетов на основе building, сгруппированных по
        зонам."""
        zones: set = self.building.zones()
        output_dir = PREPARED_DIR / self.building.name / f"{time_from}_{time_to}"
        os.makedirs(output_dir, exist_ok=True)

        print("⚙️ Запущен процесс создания датасетов с температурными показателями.")
        print(
            f"Здание поделено на {len(zones)} зон(ы). Выделяются следующие зоны {zones}"
        )
        print(f"Будет создано столько же файлов в директории. {output_dir}")

        for zone in zones:
            dataset_name = "zone_%s.pkl.gz" % zone
            units = self.building.units(zone_n=zone)
            print(f"📃 Создаем датасет {dataset_name}", end="\x1b[1K\r")
            start_time = time.monotonic()
            ds = self._merge_temps(units, time_from, time_to)
            ds.to_pickle(output_dir / dataset_name)
            print(
                f"🕑 Создали и сохранили {dataset_name} за {time.monotonic()-start_time:.2f} сек"
            )
        print("🎉 Все, завершили.")
