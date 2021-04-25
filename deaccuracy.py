import emoji
import os
import pandas as pd
import pytz

from .building import Building
from .constants import *


class DeAccuracy:
    def __init__(self, building: Building):
        self.building: Building = building
        self.working_dir = building.working_dir
        self.output_dir = BIN_DIR / building.name
        os.makedirs(self.output_dir, exist_ok=True)

    def reduce_csv(self, csv_filename):
        output = BIN_DIR / self.building.name / (csv_filename + ".pkl.gz")
        if os.path.exists(output):
            print(emoji.emojize(f":memo: {csv_filename} File already processed."))
        elif os.path.exists(self.working_dir / csv_filename):
            print(emoji.emojize(f":eyes: {csv_filename} Reading file..."))
            df = pd.read_csv(self.working_dir / csv_filename)
            print(emoji.emojize(f":crossed_fingers: {csv_filename} Rounding values..."))
            df.value = df.value.round(1)

            print(emoji.emojize(f":stopwatch: {csv_filename} Rounding timestamps..."))
            try:
                df.time = df.time.apply(
                    lambda x: pd.Timestamp(x, tz="America/New_York").round(
                        freq="T", ambiguous=True, nonexistent="shift_forward"
                    )
                )
            except pytz.exceptions.NonExistentTimeError as e:
                with open("time_errors.txt", "a") as f:
                    f.write(f"{csv_filename}, - {e} \n\n")
                    return

            print(
                emoji.emojize(f":right_arrow_curving_up: {csv_filename} Exporting...")
            )
            df.to_pickle(output)
            print(emoji.emojize(f":check_mark_button: {csv_filename} Saved!"))
        else:
            print(
                emoji.emojize(
                    f":prohibited: Cannot read {csv_filename}! It doesn't exist!"
                )
            )
