from __future__ import annotations

import getpass
import json
import os


class Binning:
    def __init__(self):
        self.runtime_filename = os.path.join(os.path.dirname(__file__), f"/tmp/runtime_binning_{getpass.getuser()}.json")
        self.default_filename = os.path.join(os.path.dirname(__file__), "default_binning.json")

        with open(self.default_filename) as df:  # read from default config
            default_data = json.load(df)
        with open(self.runtime_filename, "w") as rf:  # write to runtime config
            json.dump(default_data, rf, indent=4)

        self.runtime_config = self.load_config(self.runtime_filename)

    def load_config(self, filename):
        with open(filename) as f:
            return json.load(f)

    def save_user_config(self):
        with open(self.runtime_filename, "w") as f:
            json.dump(self.runtime_config, f, indent=4)

    def set(self, key, value):
        # Only updates the user configuration
        self.runtime_config[key] = value
        self.save_user_config()

    def reset(self, key):
        # Only updates the user configuration
        self.runtime_config[key] = self.load_config(self.default_filename)[key]
        self.save_user_config()

    def get(self, key):
        # Retrieve from the user configuration
        return self.runtime_config.get(key, None)

    def reset_all(self):
        # Only updates the user configuration
        self.runtime_config = self.load_config(self.default_filename)
        self.save_user_config()
