import csv
import matplotlib
import matplotlib.cm as cm
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import os
import pandas as pd
import pathlib
import sklearn
import sys
import time

from colour import Color
from datetime import datetime, timezone, date
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sys import exit
from enum import IntEnum

# Random state.
RS = 14545
prepare_data_per_days = None
days = []


class CalcMethod(IntEnum):
    DELAUNAY = 0  # Метод Делоне.
    SUBSEQUENT = 1  # Метод последовательной триангуляции.


def rolling_filter(arr: np.ndarray, win, func, axis):
    n_arr = np.ndarray(arr.shape)
    s = len(arr)
    for i in range(s):
        start = i - win + 1
        if start < 0:
            start = 0
        a = arr[start : i + 1, :]
        n_arr[i] = func(a, axis)
    return n_arr


def scatter(coord: np.ndarray, V: np.ndarray, T: np.ndarray, plot_mode: int):
    """
    Метод для формирования графика на основе некоторых данных
    :param coord: координаты точек - двумерный массив (x, y значений)
    :param V: массив атак (0 - нет атаки, иначе тип атаки)
    :param T: массив временных отметок (timestamp)
    :param plot_mode: тип графика: 0 - точки и линии, 1 - только линии, 2 - только точки
    :return: f: фигура, ax: график, scc: массив коллекций
    """
    TS = np.divide(np.subtract(T, T.min()), (T.max() - T.min()))
    scc = []

    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("RdBu_r", len(T)))

    # We create a scatter plot.
    f = plt.figure(figsize=(21, 18))
    ax = plt.subplot(aspect="equal")
    norm = plt.Normalize(0, 1)

    if plot_mode == 0 or plot_mode == 1:
        points = np.array([coord[:, 0], coord[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments, cmap="gnuplot", norm=plt.Normalize(0, 1), zorder=-1
        )
        # Set the values used for colormapping
        lc.set_array(TS)
        lc.set_linewidth(2)
        sc = ax.add_collection(lc)
        # sc = ax.plot(x[:, 0], x[:, 1], c=dp, zorder=-1, lw=1)
        scc.append(sc)

    if plot_mode == 0 or plot_mode == 2:
        markers = [".", "v", "^", "*", "X", "s", "D"]
        dx = coord[V == 0, 0]
        dy = coord[V == 0, 1]
        dp = TS[V == 0]
        sc = ax.scatter(
            dx, dy, c=dp, label="N", marker=markers[0], norm=norm, cmap="gnuplot"
        )
        scc.append(sc)
        for i in range(V.min() + 1, V.max() + 1):
            if i in V:
                dx = coord[V == i, 0]
                dy = coord[V == i, 1]
                dp = TS[V == i]
                sc = ax.scatter(
                    dx,
                    dy,
                    lw=0.5,
                    s=50,
                    marker=markers[i],
                    alpha=0.8,
                    norm=norm,
                    label=i,
                    c=dp,
                    edgecolors="red",
                    cmap="gnuplot",
                )
                scc.append(sc)

    plt.autoscale()
    ax.axis("tight")
    plt.legend()

    max_ticks = 50
    ticks = [i / (max_ticks - 1) for i in range(0, max_ticks)]
    ticks_lab = []
    T_count = len(T)
    for i in range(0, max_ticks):
        c_time_index = int((i / (max_ticks - 1)) * (T_count - 1))
        if c_time_index >= T_count:
            c_time_index = T_count - 1
        dt = datetime.fromtimestamp(T[c_time_index])
        ticks_lab.append(dt.strftime("%H:%M:%S"))

    cbar = plt.colorbar(cm.ScalarMappable(cmap="gnuplot"), ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks_lab)

    return f, ax, scc


def prepare_data_per_days_0(coord: np.ndarray, V: np.ndarray, T: np.ndarray):
    coord_arr = []
    v_arr = []
    t_arr = []

    prev_day = datetime.fromtimestamp(T[0])
    nv = np.empty(0, dtype=np.int32)
    nt = np.empty(0, dtype=np.int32)
    nc = np.ndarray((0, 2), dtype=np.float32)

    for i in range(len(T)):
        cur_day = datetime.fromtimestamp(T[i])
        if cur_day.date().day != prev_day.date().day:
            coord_arr.append(nc)
            v_arr.append(nv)
            t_arr.append(nt)
            nv = np.empty(0, dtype=np.int32)
            nt = np.empty(0, dtype=np.int32)
            nc = np.ndarray((0, 2), dtype=np.float32)

        nc = np.concatenate((nc, [coord[i]]))
        nv = np.append(nv, V[i])

        cur_time = cur_day.time()
        nt = np.append(
            nt, cur_time.hour * 3600 + cur_time.minute * 60 + cur_time.second
        )
        prev_day = cur_day

    coord_arr.append(nc)
    v_arr.append(nv)
    t_arr.append(nt)

    return coord_arr, v_arr, t_arr


def prepare_data_per_days_1(coord: np.ndarray, V: np.ndarray, T: np.ndarray):
    coord_arr = []
    v_arr = []
    t_arr = []

    prev_time = int((T[0] / 1000) / 60) % 10
    nv = np.empty(0, dtype=np.int32)
    nt = np.empty(0, dtype=np.int32)
    nc = np.ndarray((0, 2), dtype=np.float32)

    for i in range(len(T)):
        cur_time = int((T[i] / 1000) / 60) % 10
        if cur_time != prev_time:
            coord_arr.append(nc)
            v_arr.append(nv)
            t_arr.append(nt)
            nv = np.empty(0, dtype=np.int32)
            nt = np.empty(0, dtype=np.int32)
            nc = np.ndarray((0, 2), dtype=np.float32)

        nc = np.concatenate((nc, [coord[i]]))
        nv = np.append(nv, V[i])

        nt = np.append(nt, T[i])
        prev_time = cur_time

    coord_arr.append(nc)
    v_arr.append(nv)
    t_arr.append(nt)

    return coord_arr, v_arr, t_arr


def scatter_by_days(
    coord: np.ndarray, V: np.ndarray, T: np.ndarray, plot_mode: int, labels, show_attack
):
    """
    Метод для формирования графика на основе некоторых данных
    :param coord: координаты точек - двумерный массив (x, y значений)
    :param V: массив атак (0 - нет атаки, иначе тип атаки)
    :param T: массив временных отметок (timestamp)
    :param plot_mode: тип графика: 0 - точки и линии, 1 - только линии, 2 - только точки
    :return: f: фигура, ax: график, scc: массив коллекций
    """

    coord_arr, v_arr, t_arr = prepare_data_per_days(coord, V, T)

    # TS = np.divide(np.subtract(T, T.min()), (T.max() - T.min()))
    scc = []

    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("RdBu_r", len(T)))

    f_arr = []
    ax_arr = []
    # scc_arr = []

    start = Color(rgb=(165 / 256, 0, 38 / 256))
    mid = Color(rgb=(233 / 256, 208 / 256, 112 / 256))
    end = Color(rgb=(49 / 256, 54 / 256, 149 / 256))

    c_start = list(start.range_to(mid, int(86400 / 2)))
    c_end = list(mid.range_to(end, int(86400 / 2)))
    c_cmap = c_start + c_end
    for i in range(len(c_cmap)):
        c_cmap[i] = c_cmap[i].hex_l
    n_cm = matplotlib.colors.ListedColormap(c_cmap)

    # n_cm = 'gnuplot'

    # We create a scatter plot.
    norm = plt.Normalize(0, 86400)

    max_ticks = 50
    ticks = [i / (max_ticks - 1) for i in range(0, max_ticks)]
    ticks_lab = []
    T_count = 86400
    for i in range(0, max_ticks):
        c_time_index = int((i / (max_ticks - 1)) * (T_count - 1))
        if c_time_index >= T_count:
            c_time_index = T_count - 1
        n_t = datetime.utcfromtimestamp(c_time_index)
        ticks_lab.append(n_t.strftime("%H:%M:%S"))

    # m_lim = 65
    markers = [".", "v", "^", "*", "X", "s", "D"]
    for current_day in range(len(t_arr)):
        f = plt.figure(figsize=(21, 18))
        ax = plt.subplot(aspect="equal")

        f_arr.append(f)
        ax_arr.append(ax)

        for ni in range(len(t_arr)):
            n_v = v_arr[ni]
            n_c = coord_arr[ni]
            n_t = t_arr[ni]

            if plot_mode == 0 or plot_mode == 1:
                points = np.array([n_c[:, 0], n_c[:, 1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                if current_day == ni:
                    lc = LineCollection(
                        segments, c=c_cmap, norm=norm, zorder=-1, alpha=0.5
                    )
                    lc.set_linewidth(2)
                else:
                    lc = LineCollection(
                        segments, c=c_cmap, norm=norm, zorder=-2, alpha=0.1
                    )
                    lc.set_linewidth(1)
                # Set the values used for colormapping
                lc.set_array(n_t)
                sc = ax.add_collection(lc)
                # sc = ax.plot(x[:, 0], x[:, 1], c=dp, zorder=-1, lw=1)
                # scc.append(sc)

            if plot_mode == 0 or plot_mode == 2:
                lb = "N"
                if ni < len(labels):
                    lb = labels[ni]

                datax = n_c[:, 0]
                datay = n_c[:, 1]
                datat = n_t
                if show_attack == 1:
                    datax = n_c[n_v == 0, 0]
                    datay = n_c[n_v == 0, 1]
                    datat = n_t[n_v == 0]

                if current_day == ni:
                    ax.scatter(
                        datax,
                        datay,
                        c=datat,
                        s=50,
                        cmap=n_cm,
                        label=f"-{lb}-",
                        marker=markers[0],
                        norm=norm,
                        zorder=50,
                    )
                else:
                    ax.scatter(
                        datax,
                        datay,
                        s=10,
                        label=f"{lb}",
                        marker=markers[0],
                        alpha=0.1,
                        zorder=ni,
                        c=datat,
                        norm=norm,
                        cmap=n_cm,
                        # c='#00ffff'
                    )

                if show_attack == 1:
                    for i in range(n_v.min() + 1, n_v.max() + 1):
                        if i in V:
                            dx = n_c[n_v == i, 0]
                            dy = n_c[n_v == i, 1]
                            dp = n_t[n_v == i]
                            if current_day == ni:
                                ax.scatter(
                                    dx,
                                    dy,
                                    lw=0.5,
                                    s=50,
                                    marker=markers[i],
                                    alpha=0.8,
                                    norm=norm,
                                    label=f"A {i} {lb}",
                                    c=dp,
                                    edgecolors="red",
                                    cmap=n_cm,
                                    zorder=100,
                                )
                            else:
                                ax.scatter(
                                    dx,
                                    dy,
                                    lw=0.5,
                                    s=10,
                                    marker=markers[i],
                                    alpha=0.1,
                                    norm=norm,
                                    label=f"A {i} {lb}",
                                    c=dp,
                                    edgecolors="black",
                                    cmap=n_cm,
                                    zorder=25,
                                )

        cbar = plt.colorbar(cm.ScalarMappable(cmap=n_cm), ticks=ticks)
        cbar.ax.set_yticklabels(ticks_lab)

        # plt.ylim([-m_lim, m_lim])
        # plt.xlim([-m_lim, m_lim])
        plt.legend()
        plt.autoscale()

    return f_arr, ax_arr, scc


def scatter_lineral(coords, labels, x_as_time, need_y_scale):
    f_arr = []
    plt_name_arr = []

    max_el = 1
    if need_y_scale:
        max_arr = []
        for el in coords:
            max_arr.append(max(el[:, 0]))
        max_el = max(max_arr)

    ticks = []
    ticks_lab = []
    if x_as_time:
        max_ticks = 13
        ticks = [86400 * i / (max_ticks - 1) for i in range(0, max_ticks)]
        T_count = 86400
        for i in range(0, max_ticks):
            c_time_index = int((i / (max_ticks - 1)) * (T_count - 1))
            if c_time_index >= T_count:
                c_time_index = T_count - 1
            n_t = datetime.utcfromtimestamp(c_time_index)
            ticks_lab.append(n_t.strftime("%H:%M:%S"))

    for el in days:
        f = plt.figure(figsize=(15, 8))
        ax = plt.subplot()
        f_arr.append(f)

        plt_name = ""

        for index in el:
            n_c = coords[index]
            lb = "N"
            if index < len(labels):
                lb = labels[index]
            plt_name = plt_name + lb

            datay = n_c[:, 0] / max_el
            datax = n_c[:, 1]

            ax.scatter(datax, datay, s=15, label=f"-{lb}-", cmap="gnuplot", zorder=50)

        if x_as_time:
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks_lab)
        ax.legend()
        ax.grid(True)
        if need_y_scale:
            plt.ylim([0, 1])

        plt_name_arr.append(plt_name)
    return f_arr, plt_name_arr


def scatter_lineral_with_attack(coords, v_data, need_y_scale):
    f_arr = []
    plt_name_arr = []

    max_el = 1
    if need_y_scale:
        max_el = max(coords[:, 0])

    f = plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    f_arr.append(f)

    plt_name = ""

    lb = "All"
    plt_name = plt_name + lb

    # points = np.array([coords[:, 0], coords[:, 1]]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #
    # lc = LineCollection(segments, cmap="gnuplot", zorder=-1)
    # lc.set_linewidth(2)
    # # Set the values used for colormapping
    # # lc.set_array(coords[:, 1])
    # ax.add_collection(lc)

    datay = coords[:, 0] / max_el
    datax = coords[:, 1] / 60000
    ax.plot(datax, datay, linewidth=0.1, markersize=2, marker="o", zorder=1)

    colors = ["red", "#ff8c00", "blue", "black", "#ff2483"]

    for i in range(v_data.min() + 1, v_data.max() + 1):
        if i in v_data:
            collection = collections.BrokenBarHCollection.span_where(
                datax,
                ymin=0,
                ymax=1,
                where=v_data == i,
                facecolor=colors[i - 1],
                alpha=0.25,
                zorder=5,
                label=f"-{i}-",
            )
            ax.add_collection(collection)

    # ax.set_yscale('log')

    ax.set_yticks([1, 0.1, 0.01, 0.001, 0.0001, 10e-16, 10e-20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.scatter(datax, datay,
    #            s=5,
    #            # label=f'-{lb}-',
    #            cmap="gnuplot",
    #            zorder=50)

    ax.legend()
    ax.grid(True)
    # if need_y_scale:
    plt.ylim([0, 1])

    plt_name_arr.append(plt_name)
    return f_arr, plt_name_arr


def read_row_callback_3(result_grid: list, row: list, iv: list, i: int):
    """
    Callback метод для чтения строки из файлов attack/*.csv
    Должна использоваться при вызове метода read_csv()

    :param result_grid: массив данных
    :param row: строка с данными
    :param iv: индексы внутри данных
    :param i: индекс строки row
    """
    for item in range(len(result_grid)):
        grid = result_grid[item]
        dg = grid["DG"]
        for min_n, max_n in dg[2]:
            if min_n <= i < max_n:
                local_i = iv[item]

                r_arr = np.zeros(0)
                for el_m in dg[1]:
                    if type(el_m) is tuple:
                        min_m, max_m = el_m
                        r_arr = np.concatenate((r_arr, np.asarray(row[min_m:max_m])))
                    else:
                        r_arr = np.concatenate((r_arr, [row[el_m]]))
                grid["D"][local_i] = r_arr
                grid["V"][local_i] = row[15]
                grid["T"][local_i] = float(row[13]) * 1000
                iv[item] += 1

                break


def read_row_callback_2(result_grid: list, row: list, iv: list, i: int):
    """
    Callback метод для чтения строки из файла bldg-MC2.csv
    Должна использоваться при вызове метода read_csv()

    :param result_grid: массив данных
    :param row: строка с данными
    :param iv: индексы внутри данных
    :param i: индекс строки row
    """
    for item in range(len(result_grid)):
        grid = result_grid[item]
        dg = grid["DG"]
        for min_n, max_n in dg[2]:
            if min_n <= i < max_n:
                local_i = iv[item]

                r_arr = np.zeros(0)
                for el_m in dg[1]:
                    if type(el_m) is tuple:
                        min_m, max_m = el_m
                        r_arr = np.concatenate(
                            (r_arr, np.asarray(row[min_m + 1 : max_m + 1]))
                        )
                    else:
                        r_arr = np.concatenate((r_arr, [row[el_m + 1]]))
                grid["D"][local_i] = r_arr
                grid["V"][local_i] = 0
                dt = row[0]
                dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
                grid["T"][local_i] = int(dt.timestamp())
                iv[item] += 1

                break


def read_row_callback_1(result_grid: list, row: list, iv: list, i: int):
    """
    Callback метод для чтения строки из файла swat2/SWaT_Dataset_Attack_v0.csv
    Должна использоваться при вызове метода read_csv()

    :param result_grid: массив данных
    :param row: строка с данными
    :param iv: индексы внутри данных
    :param i: индекс строки row
    """
    for item in range(len(result_grid)):
        grid = result_grid[item]
        dg = grid["DG"]
        if (i >= dg[3]) and (i < dg[4]):
            local_i = iv[item]
            grid["D"][local_i] = np.asarray(row[dg[1] + 2 : dg[2] + 2])
            grid["V"][local_i] = int(row[1])
            dt = row[0]
            dt = datetime.strptime(dt, " %d/%m/%Y %I:%M:%S %p")
            grid["T"][local_i] = int(dt.timestamp())
            iv[item] += 1


def read_row_callback_0(result_grid: list, row: list, iv: list, i: int):
    """
    Callback метод для чтения строки из файла SWaT.A4 _ A5_Jul 2019\\SWaT_dataset_Jul 19 v2.csv
    Должна использоваться при вызове метода read_csv()

    :param result_grid: массив данных
    :param row: строка с данными
    :param iv: индексы внутри данных
    :param i: индекс строки row
    """
    for item in range(len(result_grid)):
        grid = result_grid[item]
        dg = grid["DG"]
        if (i >= dg[3]) and (i < dg[4]):
            local_i = iv[item]
            grid["D"][local_i] = np.asarray(row[dg[1] + 4 : dg[2] + 4])
            grid["V"][local_i] = int(row[3])
            dt = row[0] + " " + row[1]
            dt = datetime.strptime(dt, "%d.%m.%Y %I:%M:%S")
            grid["T"][local_i] = int(dt.timestamp())

            # if isinstance(dt, str):
            #     dt = datetime.strptime(dt, '%d.%m.%Y')
            # t = row[1]
            # if isinstance(t, str):
            #     t = datetime.strptime(t, '%H:%M:%S').time()
            # grid['T'][local_i] = int(datetime.combine(dt.date(), t).timestamp())
            iv[item] += 1


def prepare_result_grid(data_grid: list):
    """
    Выполняет подготовку массива данных для отображения на основе кастомизируемых данных data_grid

    :param data_grid: данные кастомизации графиков
    :return: result_grid: массивы данных,
             iv: индексты внутри массивов данных,
             min_n_start: минимальный требуемый индекс для чтения,
             max_n_end: максимальный требуемый индекс для чтения,
    """
    result_grid = []
    min_arr_n = []
    max_arr_n = []
    for item in data_grid:
        m = 0
        n = 0
        for el in item[1]:
            if type(el) is tuple:
                col_interval_start, col_interval_end = el
                m += col_interval_end - col_interval_start
            else:
                m += 1

        for row_interval_start, row_interval_end in item[2]:
            n += row_interval_end - row_interval_start
            max_arr_n.append(row_interval_end)
            min_arr_n.append(row_interval_start)

        v = np.zeros(n, dtype=np.int32)
        t = np.zeros(n, dtype=np.int32)
        d = np.zeros((n, m))
        result_grid.append({"V": v, "T": t, "D": d, "DG": item})

    iv = [0 for el in data_grid]
    min_n_start = min(min_arr_n)
    max_n_end = max(max_arr_n)

    return result_grid, iv, min_n_start, max_n_end


def read_csv(
    data_grid,
    filename: str,
    read_row_func,
    delimiter,
    first_row_handler=None,
    start_reading_row=1,
):
    """
    Метод чтения произвольных csv файлов и формирования массивов данных для построения графиков

    :param data_grid: данные кастомизации графиков
    :param filename: путь к файлу с данными
    :param read_row_func: callback-метод чтения(парсинга) одной строки из файла
    :param delimiter: знак-разделитель значений
    :param start_reading_row: индекс строки, начиная с которой будут данные для чтения
    :return:
    """
    add_data = None
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        i = 0  # dsf
        j = 0

        for row in reader:
            if i == 0:
                if first_row_handler is not None:
                    add_data = first_row_handler(row)
                    if add_data is not None and len(add_data) > 0:
                        data_grid = data_grid + add_data
                result_grid, iv, min_n, max_n = prepare_result_grid(data_grid)

            if i >= start_reading_row:
                if min_n <= j < max_n:
                    read_row_func(result_grid, row, iv, j)
                j += 1
            i += 1

        print("Ok reading")
        return result_grid, add_data


def write_points_csv(
    coords: np.ndarray, v_data: np.ndarray, t_data: np.ndarray, filename: str
):
    """
    Метод записи точек в файл csv
    :param coords: координаты точек - двумерный массив (x, y значений)
    :param v_data: массив атак (0 - нет атаки, иначе тип атаки)
    :param t_data: массив временных отметок (timestamp)
    :param filename: имя (путь) нового файла
    :return:
    """
    print("Writing begin")
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        writer.writerow(
            ["id_record", "Date/Time", "x", "y", "timestamp_ms", "anomaly_class"]
        )
        i = 0
        for coord in coords:
            writer.writerow(
                [
                    i,
                    datetime.utcfromtimestamp(t_data[i] / 1000),
                    coord[0],
                    coord[1],
                    t_data[i],
                    v_data[i],
                ]
            )
            i += 1
        print("Ok writing")


def import_from_csv(filename):
    with open(filename) as f:

        max_size = 100000

        reader = csv.reader(f, delimiter=",")
        nv = np.empty(max_size, dtype=np.int32)
        nt = np.empty(max_size, dtype=np.int32)
        nc = np.ndarray((max_size, 2), dtype=np.float32)

        reader.__next__()
        print("Start import")

        i = 0
        total = 0
        t1 = time()
        t2 = time()
        for row in reader:
            nt[total] = int(row[4])
            cc = np.array([float(row[2]), float(row[3])])
            nc[total] = cc
            nv[total] = int(row[5])

            i += 1
            total += 1
            if i >= max_size:
                nt = np.append(nt, np.empty(max_size, dtype=np.int32))
                nv = np.append(nv, np.empty(max_size, dtype=np.int32))
                nc = np.concatenate((nc, np.ndarray((max_size, 2), dtype=np.float32)))
                # nc = np.append(nc, np.ndarray((50000, 2), dtype=np.float32))

                i = 0
                print(f"{time() - t1} {total}")
                t1 = time()

        print(f"{time() - t2} Ok import from " + filename)
        return nc[0:total], nv[0:total], nt[0:total]


def pretty_triang_plot(
    coords: np.ndarray,
    v_data: np.ndarray,
    t_data: np.ndarray,
    file_name,
    labels,
    images_path,
):
    dir_name = f"{images_path}/tr/" + file_name
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    triangul_arr = triangulate_per_day(coords, v_data, t_data)
    f_arr, plot_names = scatter_lineral(triangul_arr, labels, True, True)
    for f_iter, f in enumerate(f_arr):
        t1 = time()
        sfn = f"{plot_names[f_iter]}"
        f.savefig(f"{dir_name}/{sfn}.png", dpi=300)
        f.show()
        print(f"Print tring plot {sfn} {time() - t1}")

        plt.close(f)


def pretty_ntriang_path(
    coords: np.ndarray,
    v_data: np.ndarray,
    t_data: np.ndarray,
    file_name,
    images_path,
    i_st,
    st,
):
    dir_name = f"{images_path}/tr/"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    triangul_arr = triangulate_path(coords, t_data, st[0])

    # def mean(arr, index, axis):
    #     return np.mean(arr, axis)
    # def mean(arr, index, axis):
    #     return np.mean(arr, axis)

    if st[1] is not None and st[2] is not None:
        triangul_arr = rolling_filter(triangul_arr, st[1], st[2], 0)

    f_arr, plot_names = scatter_lineral_with_attack(triangul_arr, v_data, True)

    for f_iter, f in enumerate(f_arr):
        t1 = time()
        sfn = f"{plot_names[f_iter]}"
        f.savefig(f"{dir_name}/{file_name}_{i_st}.png", dpi=300)
        f.show()
        print(f"Print tring plot {sfn} {time() - t1}")

        plt.close(f)


def pretty_plot(
    coords: np.ndarray,
    v_data: np.ndarray,
    t_data: np.ndarray,
    file_name,
    labels,
    images_path,
    one_day,
):
    if one_day:
        # график только для одного дня
        f, ax, scc = scatter(coords, v_data, t_data, 2)
        f.savefig(f"{images_path}/{file_name}.png", dpi=250)
        f.show()
    else:
        # графики нескольких дней
        type = 2  # 0 - и точки, и линии, 1 - линии, 2 - точки
        show_attack = 0  # 0 - не выделять атаки, 1 - выделять атаки

        f_arr, ax_arr, sc_arr = scatter_by_days(
            coords, v_data, t_data, type, labels, show_attack
        )
        f_iter = 0

        dir_name = f"{images_path}/" + file_name
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        for f in f_arr:
            t1 = time()
            sfn = f"{file_name}_{type}{show_attack}_{f_iter}"
            f.savefig(f"{dir_name}/{sfn}.png", dpi=300)
            f.show()
            print(f"Print plot {sfn} {time() - t1}")
            f_iter += 1

            plt.close(f)


def calc_sq_triangle(vertex_arr):
    s = 1
    m_type = len(vertex_arr)
    if m_type == 2:
        s = pow(
            pow((vertex_arr[1][0] - vertex_arr[0][0]), 2)
            + pow((vertex_arr[0][1] - vertex_arr[1][1]), 2),
            0.5,
        )
    elif m_type == 3:
        p1 = vertex_arr[0]
        p2 = vertex_arr[1]
        p3 = vertex_arr[2]
        s = (p1[0] - p3[0]) * (p2[1] - p3[1]) - ((p1[1] - p3[1]) * (p2[0] - p3[0]))
        s = abs(s) * 0.5
    elif m_type == 4:
        p1 = vertex_arr[0]
        p2 = vertex_arr[1]
        p3 = vertex_arr[2]
        p4 = vertex_arr[3]
        s = (
            p1[0] * p2[1]
            + p2[0] * p3[1]
            + p3[0] * p4[1]
            + p4[0] * p1[1]
            - p2[0] * p1[1]
            - p3[0] * p2[1]
            - p4[0] * p3[1]
            - p1[0] * p4[1]
        )
        s = abs(s) * 0.5
    return s


def triangulate_per_day(coords: np.ndarray, v_data: np.ndarray, t_data: np.ndarray):
    coord_arr, v_arr, t_arr = prepare_data_per_days(coords, v_data, t_data)

    tr_arr = []
    for k in range(len(coord_arr)):
        el = coord_arr[k]
        y_arr = np.transpose(t_arr[k][0 : len(el) - 2])
        n_s_arr = []
        for i in range(len(el) - 2):
            # p1 = el[i]
            # p2 = el[i+1]
            # p3 = el[i+2]
            # s = (p1[0] - p3[0])*(p2[1]-p3[1])-((p1[1]-p3[1])*(p2[0]-p3[0]))
            # s = abs(s) * 0.5
            n_s_arr.append(calc_sq_triangle((el[i], el[i + 1], el[i + 2])))

        temp_arr = np.array(n_s_arr)
        temp_arr = np.transpose(temp_arr)
        t = np.array((temp_arr, y_arr))
        t = np.transpose(t)
        t = t.reshape((len(temp_arr), 2))

        tr_arr.append(t)

    return tr_arr


def triangulate_path(coords: np.ndarray, t_data: np.ndarray, sizes):
    y_arr = np.transpose(t_data[0 : len(coords) - (sizes - 1)])
    n_s_arr = []
    for i in range(len(coords) - (sizes - 1)):
        cd = []
        for j in range(sizes):
            cd.append(coords[i + j])
        n_s_arr.append(calc_sq_triangle(cd))

    temp_arr = np.array(n_s_arr)
    temp_arr = np.transpose(temp_arr)
    t = np.array((temp_arr, y_arr))
    t = np.transpose(t)
    t = t.reshape((len(temp_arr), 2))

    return t


def t_path_per_day(coords: np.ndarray, v_data: np.ndarray, t_data: np.ndarray):
    coord_arr, v_arr, t_arr = prepare_data_per_days(coords, v_data, t_data)

    tr_arr = []
    for k in range(len(coord_arr)):
        el = coord_arr[k]

        k_size = 0
        t_path = 0
        for i in range(len(el) - 2):
            t_path += calc_sq_triangle((el[i], el[i + 1], el[i + 2]))
            k_size += 1

        tr_arr.append((t_path, k_size))

    return tr_arr


def pretty_t_path(
    coords: np.ndarray,
    v_data: np.ndarray,
    t_data: np.ndarray,
    file_name,
    labels,
    images_path,
):
    dir_name = f"{images_path}/tr/" + file_name
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    t_path_arr = t_path_per_day(coords, v_data, t_data)
    sq_arr = [s for s, _ in t_path_arr]

    f = plt.figure(figsize=(7.5, 4))
    ax = plt.subplot()

    norm_sq_arr = sq_arr / max(sq_arr)
    print(sq_arr)
    with open("data.csv", mode="w") as data_csv:
        data_csv = csv.writer(
            data_csv, delimiter=";", quotechar="'", quoting=csv.QUOTE_MINIMAL
        )

        data_csv.writerow(sq_arr)

    xticks = [i for i in range(0, len(labels))]
    ax.scatter(xticks, norm_sq_arr, zorder=50)
    for i, sq in enumerate(norm_sq_arr):
        ax.annotate("{:0.2f}".format(sq), xy=(xticks[i] + 0.1, norm_sq_arr[i] + 0.01))

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.grid(True)
    plt.legend(["Total path"])
    plt.ylim([0, 1.1])

    t1 = time()
    sfn = f"t_path"
    f.savefig(f"{dir_name}/{sfn}.png", dpi=300)
    f.show()
    print(f"Print sq triangles plot {sfn} {time() - t1}")

    plt.close(f)


def delaunay_per_day(coordinates: np.ndarray, v_data: np.ndarray, t_data: np.ndarray):
    coord_arr, v_arr, t_arr = prepare_data_per_days(coordinates, v_data, t_data)
    tr_arr = [(coord_arr[i], Delaunay(el)) for i, el in enumerate(coord_arr)]
    return tr_arr


def pretty_delaunay(
    pca_data: np.ndarray,
    v_data: np.ndarray,
    t_data: np.ndarray,
    file_name,
    labels,
    images_path,
):
    """Функция использует метод Делоне"""

    delaunay_arr = delaunay_per_day(pca_data, v_data, t_data)

    sq_arr = []
    del_count = []
    for i, (xy, delaunay) in enumerate(delaunay_arr):
        day_sq = 0
        for triang in delaunay.simplices:
            tr_coods = [xy[i] for i in triang]
            sq = calc_sq_triangle(tr_coods)
            day_sq += sq
        sq_arr.append(day_sq)
        del_count.append(len(delaunay.simplices))

    f = plt.figure(figsize=(7.5, 4))
    ax = plt.subplot()

    norm_sq_arr = sq_arr / max(sq_arr)
    mean_sq_arr = [sq / dc for sq, dc in zip(sq_arr, del_count)]
    print(sq_arr)
    print(mean_sq_arr)
    with open("data.csv", mode="w") as data_csv:
        data_csv = csv.writer(
            data_csv, delimiter=";", quotechar="'", quoting=csv.QUOTE_MINIMAL
        )

        data_csv.writerow(sq_arr)
        data_csv.writerow(mean_sq_arr)

    norm_mean_sq_arr = mean_sq_arr / max(mean_sq_arr)
    xticks = [i for i in range(0, len(labels))]
    ax.scatter(xticks, norm_sq_arr, zorder=50)
    for i, sq in enumerate(norm_sq_arr):
        ax.annotate("{:0.2f}".format(sq), xy=(xticks[i] + 0.1, norm_sq_arr[i] + 0.01))

    ax.scatter(xticks, norm_mean_sq_arr, zorder=20, marker="*")
    # for i, sq in enumerate(norm_mean_sq_arr):
    #     ax.annotate('{:0.2f}'.format(sq),
    #                 xy=(xticks[i] - 0.2, norm_mean_sq_arr[i] + 0.01))

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.grid(True)
    plt.legend(["Total sq.", "Mean sq.e"])
    plt.ylim([0, 1.1])

    t1 = time()
    sfn = f"mean_sq_triangles"
    f.savefig(f"{dir_name}/{sfn}.png", dpi=300)
    f.show()
    print(f"Print sq triangles plot {sfn} {time() - t1}")

    plt.close(f)


def pretty_export(
    coords: np.ndarray, v_data: np.ndarray, t_data: np.ndarray, file_name, export_path
):
    t1 = time()
    write_points_csv(coords, v_data, t_data, f"{export_path}/{file_name}.csv")
    print(f"Export data {file_name} {time() - t1}")


def first_row_handler_0(row):
    list_e = ["F_1_Z_3", "F_3_Z_5", "F_3_Z_9"]

    i_start = 13
    i = 0
    data = {}
    for i in range(i_start, len(row)):
        el = row[i]
        name = el.split()[0]
        name = name.split(":")[0]
        if name[-1] == ":":
            name = name[0:-1]
        if name in data:
            data[name].append(i - 1)
        else:
            data[name] = [i - 1]
        i += 1

    data_grid = []
    for key in data.keys():
        if (key in list_e and len(list_e) > 0) or len(list_e) == 0:
            arr = data[key]
            nkey = "Day 31-13 " + key
            data_grid.append([nkey, arr, [(0, 4031)]])

    return data_grid


def main(args):
    global prepare_data_per_days, days

    print(f"Start Program with args: {args}")
    path = os.getcwd()
    print("The current working directory is %s" % path)

    # Выбор метода
    # 0 - МЕТОД РАСЧЕТА МЕТРИКИ НА ОСНОВЕ ОБЩЕЙ ПЛОЩАДИ ТРЕУГОЛЬНИКОВ, ПОЛУЧЕННЫХ ТРИАНГУЛЯЦИЕЙ ДЕЛОНЕ
    # 1 - МЕТОД ПОСЛЕДОВАТЕЛЬНОЙ ТРИАНГУЛЯЦИИ для оценки поведения системы на интервале
    method = args["method"]
    print(f"Ⓜ Метод - {method}")

    # fname - os.DirEntry
    fnames = os.scandir(args["import_dir"])
    for fname in fnames:

        df = pd.read_pickle(fname.path)

        print(f"👀 Посмотрим датафрейм {fname.name}")

        # Пока файл будет рандомное имя иметь, потому что я не понял по какому принципу
        # файлы называются в исходнике
        file_name = "%s %s" % (
            time.strftime("%a, %d %b %H:%M"),
            np.random.randint(1000),
        )

        # Вычислим PCA по данным датасета
        print(f"1️⃣ Считаем PCA на массиве данных размером {df.shape}")
        pca_result = PCA(n_components=2).fit_transform(df)

        # if method == CalcMethod.DELAUNAY:
        #     pretty_delaunay(pca_result, None, None, file_name, [], images_path)

        # for i_st, st in enumerate(settings):
        #     ###### t-sne
        #     # блок обработки данных алгоритмом TSNE
        #     # нужно выбрать либо TSNE, либо PCA
        #     # digits_proj = TSNE(random_state=RS,
        #     #                    n_iter=1000,
        #     #                    perplexity=30).fit_transform(grid['D'])
        #     # print(f'TSNE for {file_name} {time() - t1}')
        #     ######

        #     ##### PCA
        #     # блок обработки данных алгоритмом PCA
        #     # нужно выбрать либо TSNE, либо PCA
        #     digits_proj = PCA(n_components=2).fit_transform(grid["D"])
        #     print(f"PCA for {file_name} {time() - t1}")
        #     ######

        #     # формирует файл с данными после сокращения размерности
        #     pretty_export(digits_proj, grid["V"], grid["T"], file_name, export_path)
        #     # строит для каждого дня отдельный график, при этом можно включить наложение остальных дней
        #     # для удобства сравнения, + временная шкала
        #     # pretty_plot(digits_proj, grid['V'], grid['T'], file_name, lbs, images_path, True)

        #     ## Для "МЕТОД РАСЧЕТА МЕТРИКИ НА ОСНОВЕ ОБЩЕЙ ПЛОЩАДИ ТРЕУГОЛЬНИКОВ, ПОЛУЧЕННЫХ ТРИАНГУЛЯЦИЕЙ ДЕЛОНЕ"
        #     if method == 0:
        #         # алгоритм поледовательного вычисления площадей треугольников делоне для каждого дня,
        #         # с построением графиков одного дня (с временем), выполняется сравнение (наложение) дней,
        #         # это конфигурируется в массиве days
        #         # - можно оценить/сравнить, насколько и чем отличаются дни
        #         # pretty_triang_plot(digits_proj, grid['V'], grid['T'], file_name, lbs, images_path)

        #         # алгоритм вычисления площади всех треугольников делоне для каждого дня, получение значения метрики
        #         # с построением графика всех дней
        #         # - можно сравнить все дни между собой
        #         # ELCONRUS
        #         pretty_delaunay(
        #             digits_proj, grid["V"], grid["T"], file_name, lbs, images_path
        #         )

        #         # алгоритм поледовательного вычисления площади треугольников делоне для каждого дня
        #         # их суммирование, получение значения метрики
        #         # с построением графика всех дней
        #         # - можно сравнить все дни между собой, выявить что-нибудь интересное
        #         # pretty_t_path(digits_proj, grid['V'], grid['T'], file_name, lbs, images_path)

        #     ## Для "МЕТОД ПОСЛЕДОВАТЕЛЬНОЙ ТРИАНГУЛЯЦИИ"
        #     if method == 1:
        #         # алгоритм последовательной триангуляции с вычислением соответсвующей метрики
        #         # на всех данных с формированием графика всех данных со шкалой времени в минутах
        #         # - можно оценить поведение системы на всем интервале времени и выявить аномальные паттерны
        #         # MECO
        #         pretty_ntriang_path(
        #             digits_proj,
        #             grid["V"],
        #             grid["T"],
        #             file_name,
        #             images_path,
        #             i_st,
        #             st,
        #         )

