import json

from .constants import *


class Building:
    """Building."""

    def __init__(self, filename):
        self.filename = filename
        self.name = filename.replace(".json", "")
        self.working_dir = SRC_DIR / self.name
        self.dict = {}
        self.variables = []
        self.references = []
        self.load()

    def load(self):

        with open(SRC_DIR / JSON_DIR / self.filename) as f:
            self.dict = json.load(f)

        global_variables = self.dict["building"]["heating_management_system"]["plc"][
            "global_variables"
        ]
        for k, v in global_variables.items():
            self.variables += v["variable"]
        for va in self.variables:
            for ref in va["refs"]["ref"]:
                self.references.append(ref)

    def floors(self):
        """Returns set of available floors."""
        units = self.dict["building"]["configuration"]["units"]["unit"]
        return set([u["floor_number"] for u in units])

    def zones(self):
        """Returns set of available zones."""
        units = self.dict["building"]["configuration"]["units"]["unit"]
        return set([u["zone_number"] for u in units])

    def units(self, floor_n=None, zone_n=None):
        """Returns list of references related to unit temps.

        NOTE: you can only use floor_n or zone_n. Using both params is not
        implemented.
        """
        res = [i for i in self.references if i["flag"] == "UNIT_TEMP"]

        if floor_n is None and zone_n is None:
            return res
        elif floor_n is not None:
            unit_ids_on_floor = list(
                filter(
                    lambda x: x["floor_number"] == floor_n,
                    self.dict["building"]["configuration"]["units"]["unit"],
                )
            )
            unit_ids_on_floor = [i["unit_id"] for i in unit_ids_on_floor]
            res = list(filter(lambda x: x["unit_id"] in unit_ids_on_floor, res))
            return res
        elif zone_n is not None:
            unit_ids_on_zone = list(
                filter(
                    lambda x: x["zone_number"] == zone_n,
                    self.dict["building"]["configuration"]["units"]["unit"],
                )
            )
            unit_ids_on_zone = [i["unit_id"] for i in unit_ids_on_zone]
            res = list(filter(lambda x: x["unit_id"] in unit_ids_on_zone, res))
            return res

    def get_csv_path(self, name_opc):
        return self.working_dir / f"{name_opc}.csv"