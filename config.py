
from collections import namedtuple


def load_config():

  constants = {}

  with open("config.h", "r") as f:
    line_list = f.readlines()

  for line in line_list:
    line = line.strip().replace(";", "").replace("\n", "")
    if line.startswith("static"):
      split = line.split(maxsplit=3)
      if len(split) >= 4:
        key, value = split[3].split("=")
        key = key.strip()
        value = value.strip()
        if key == "use_amp":
          value = str(bool(value))
        for k, v in constants.items():
          if k in value:
            value = value.replace(k, str(v))
        eval_value = eval(value)
        if type(eval_value) is float or type(eval_value) is int:
          constants[key] = int(eval_value) if eval_value > 1 else eval_value
        else:
          constants[key] = eval_value
    elif line.startswith("inline"):
      split = line.split(maxsplit=4)
      key, value = split[4].split("=")
      constants[key.strip()] = eval(value.strip())

  return namedtuple("Config", sorted(constants))(**constants)


Config = load_config()
