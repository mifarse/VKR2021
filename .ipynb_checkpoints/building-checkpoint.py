from constants import *

class Building:
    def __init__(self, filename):
        self.filename = filename
        self.dict = {}
        self.variables = []
        self.references = []
        self.load()
        
    def load(self):
        with open(SRC_DIR / JSON_DIR / self.filename) as f:
            self.dict = json.load(f)
            
        global_variables = self.dict["building"]["heating_management_system"]["plc"]["global_variables"]
        for k,v in global_variables.items():
            self.variables += v["variable"]
        for va in all_vars:
            for ref in va["refs"]["ref"]:
                self.references.append(ref)

    def hello(self):
        print("hello")