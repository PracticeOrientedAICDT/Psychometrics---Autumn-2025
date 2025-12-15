import random

def pick_value(rule):
    """Pick a value from rule['left'] or rule['right']."""
    v = rule
    if isinstance(v, dict):
        # range
        if "min" in v and "max" in v:
            return random.randint(v["min"], v["max"])
        # minlimit, maxlimit ignored here
        if "minlimit" in v:
            return random.randint(v["minlimit"], v["max"])
    elif isinstance(v, list):
        return random.choice(v)
    else:
        return v  # literal?
        

def compute_answer(op, left, right):
    if op == "addition":
        return left + right
    if op == "subtraction":
        return left - right
    if op == "multiplication":
        return left * right
    if op == "division":
        return left // right  # these rules always give clean integer divisions
    if op == "fraction":
        # left is something like "1/3"
        num, den = left.split("/")
        num = int(num)
        den = int(den)
        # fraction of 'right'
        return (num/den) * right
    if op == "percentage":
        return (left/100) * right

    return None


# ============================================================
# FULL LEVEL FUNCTION TABLE (1–30)
# ============================================================

LEVEL_FUNCTIONS = {
    1:  {"operation": "addition",      "left": [2,3,4,5,6,7],       "right":[2,3,4,5,6,7,8],   "maxlimit":9},
    2:  {"operation": "subtraction",   "left": [4,5,6,7,8,9],       "right":[2,3,4,5,6,7,8],   "maxlimit":9},
    3:  {"operation": "addition",      "left": [2,3,4,5,6,7,8],     "right":[2,3,4,5,6,7,8],   "minlimit":10},
    4:  {"operation": "subtraction",   "left": {"min":10,"max":16}, "right":[2,3,4,5,6,7,8,9], "maxlimit":9},
    5:  {"operation": "multiplication","left":[2,3,4,5,10],         "right":[2,3,4,5,6,8]},
    6:  {"operation": "division",      "left":[2,3,4,5,10],         "right":[2,3,4,5,6,8]},

    7:  {"operation":"addition",       "left":{"min":11,"max":30},  "right":{"min":1,"max":9}},
    8:  {"operation":"subtraction",    "left":[20,30,40,50,60,70,80,90],"right":{"min":1,"max":9},"minlimit":12},
    9:  {"operation":"fraction",       "left":["1/4","1/2","1/3","1/5"],"right":[2,3,4,5,6,7,8,9]},

    10: {"operation":"addition",       "left":{"min":11,"max":60},  "right":{"min":11,"max":30}},
    11: {"operation":"percentage",     "left":[10,20,50],           "right":[2,3,4,5,6,7,8,9]},
    12: {"operation":"subtraction",    "left":{"min":21,"max":70},  "right":{"min":11,"max":30}},
    13: {"operation":"addition",       "left":[20,30,40,50,60,70,80,90],"right":{"min":11,"max":30},"decimal":10},

    14: {"operation":"multiplication", "left":[4,6,7,8,9,11,12],    "right":{"min":3,"max":12}},
    15: {"operation":"addition",       "left":{"min":111,"max":899},"right":{"min":11,"max":50}},
    16: {"operation":"percentage",     "left":[10,20,25,40],        "right":[4,5,6,8,10,12,15]},
    17: {"operation":"subtraction",    "left":[20,30,40,50,60,70,75,80,90],"right":{"min":11,"max":30},"decimal":10},

    18: {"operation":"fraction",       "left":["1/3","2/3","1/5","2/5","3/5","4/5","1/8","3/8"],"right":{"min":6,"max":20}},
    19: {"operation":"subtraction",    "left":{"min":111,"max":899},"right":{"min":11,"max":60}},

    20: {"operation":"division",       "left":[4,6,7,8,9,11,12],    "right":{"min":3,"max":12}},
    21: {"operation":"addition",       "left":{"min":111,"max":899},"right":{"min":11,"max":50},"decimal":10},
    22: {"operation":"multiplication", "left":{"min":7,"max":19},   "right":{"min":4,"max":20}},
    23: {"operation":"subtraction",    "left":{"min":111,"max":899},"right":{"min":11,"max":60},"decimal":10},

    24: {"operation":"division",       "left":{"min":7,"max":19},   "right":{"min":4,"max":20}},

    25: {"operation":"fraction",       "left":["1/5","2/5","3/5","4/5","1/6","5/6","1/8","3/8","5/8","7/8"],
                                        "right":{"min":11,"max":50}},
    26: {"operation":"percentage",     "left":[15,30,40,60,70,80,85],
                                        "right":[30,40,50,60,70,80,90]},

    27: {"operation":"addition",       "left":{"min":111,"max":899},"right":{"min":111,"max":899}},
    28: {"operation":"subtraction",    "left":{"min":212,"max":899},"right":{"min":111,"max":899}},
    29: {"operation":"addition",       "left":{"min":111,"max":899},"right":{"min":111,"max":899},"decimal":100},
    30: {"operation":"subtraction",    "left":{"min":212,"max":899},"right":{"min":111,"max":899},"decimal":100},
}

