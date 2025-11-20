import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import random
from calculations import LEVEL_FUNCTIONS, pick_value, compute_answer
from utils import clamp

# engine.py
import time, random
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# --- CONFIG ---
ALLOW_NEGATIVES = False  # set True to allow 3 - 6 style negatives
WINDOW_N = 3             # moving window size
MAX_DIFFICULTY = 30      # from spec's table

# --- helpers to sample rule values (range dict or list) ---
def pick_value(v):
    """Pick either from list or from {min,max} dict."""
    if isinstance(v, dict):
        return random.randint(v["min"], v["max"])
    return random.choice(v)

def enforce_bounds(res, minlimit=None, maxlimit=None):
    if minlimit is not None and res < minlimit:
        return False
    if maxlimit is not None and res > maxlimit:
        return False
    return True

def compute_answer(op: str, left: int, right: int) -> str:
    if op == "addition":
        return str(left + right)
    if op == "subtraction":
        return str(left - right)
    if op == "multiplication":
        return str(left * right)
    if op == "division":
        # integer division cases in the spec are exact; guard divide-by-zero
        return str(left // right) if right != 0 else "0"
    if op == "percentage":
        # “left percent of right” ⇒ (left/100)*right, round to int like the original game
        return str(int(round((left / 100.0) * right)))
    if op == "fraction":
        # left is a string like '1/3', '3/5' and right is an int ⇒ a/b of right
        a, b = left.split("/")
        val = (int(a) / int(b)) * right
        return str(int(round(val)))
    return "0"

def postprocess_operands(op, left, right):
    if op == "subtraction" and not ALLOW_NEGATIVES and left < right:
        left, right = right, left
    return left, right

def _to_int_if_integer_like(x):
    """Return int(x) if x is an integer-valued number/string; otherwise return x unchanged."""
    try:
        xf = float(x)
        if xf.is_integer():
            return int(xf)
    except Exception:
        pass
    return x

def sample_operands(rule):
    op = rule["operation"]
    left = pick_value(rule["left"])
    right = pick_value(rule["right"])

    # --- MULTIPLICATION: integers only (no decimals) ---
    if op == "multiplication":
        # Try to coerce integer-like values (e.g., 3.0 -> 3) and resample if needed
        for _ in range(30):
            L = _to_int_if_integer_like(left)
            R = _to_int_if_integer_like(right)
            if isinstance(L, int) and isinstance(R, int):
                left, right = L, R
                break
            # resample until both are ints
            left = pick_value(rule["left"])
            right = pick_value(rule["right"])
        # final safety cast (in case sources are floats like 7.0)
        left = int(left) if not isinstance(left, int) and float(left).is_integer() else left
        right = int(right) if not isinstance(right, int) and float(right).is_integer() else right

    # --- DIVISION: enforce exact integer division (no remainder, no /0) ---
    elif op == "division":
        for _ in range(30):
            # coerce int-like
            L = _to_int_if_integer_like(left)
            R = _to_int_if_integer_like(right)
            if isinstance(L, int) and isinstance(R, int) and R != 0 and L % R == 0:
                left, right = L, R
                break
            left = pick_value(rule["left"])
            right = pick_value(rule["right"])
        # last resort: force exact
        if not isinstance(right, int) or right == 0:
            right = 1
        left = right * random.randint(1, 10)

    # --- PERCENTAGE / FRACTION: keep integers on the numeric side ---
    elif op == "percentage":
        left = int(_to_int_if_integer_like(left))   # percent value like 25
        right = int(_to_int_if_integer_like(right)) # base like 80

    elif op == "fraction":
        # Fractions are strings like '1/3' for left; right must be int
        right = int(_to_int_if_integer_like(right))

    # --- Enforce optional min/max result bounds (from spec) ---
    if "minlimit" in rule or "maxlimit" in rule:
        for _ in range(20):
            L = pick_value(rule["left"])
            R = pick_value(rule["right"])

            # apply the same coercions for the trial operands
            if op == "multiplication":
                L = _to_int_if_integer_like(L)
                R = _to_int_if_integer_like(R)
                if not (isinstance(L, int) and isinstance(R, int)):
                    continue  # not acceptable, try again
            elif op == "division":
                L = _to_int_if_integer_like(L)
                R = _to_int_if_integer_like(R)
                if not (isinstance(L, int) and isinstance(R, int)):
                    continue
                if R == 0 or L % R != 0:
                    continue
            elif op == "percentage":
                L = int(_to_int_if_integer_like(L))
                R = int(_to_int_if_integer_like(R))
            elif op == "fraction":
                R = int(_to_int_if_integer_like(R))

            val = compute_numeric(op, L, R)
            if enforce_bounds(val, rule.get("minlimit"), rule.get("maxlimit")):
                return L, R

    return left, right


def compute_numeric(op, left, right):
    if op == "addition":        return left + right
    if op == "subtraction":     return left - right
    if op == "multiplication":  return left * right
    if op == "division":        return left // right if right != 0 else 0
    if op == "percentage":      return int(round((left / 100.0) * right))
    if op == "fraction":
        a, b = str(left).split("/")
        return int(round((int(a) / int(b)) * right))
    return 0

def difficulty_for_level(level: int, balloon_index: int, n: int = WINDOW_N) -> int:
    """Spec window:
       L1: [1,1,1]; L2: [2,1,2]; L>=3: [L,L-1,L-2] cycling by balloon_index.
       Clamp to [1, MAX_DIFFICULTY].
    """
    if level <= 1:
        choices = [1, 1, 1]
    elif level == 2:
        choices = [2, 1, 2]
    else:
        choices = [level, level-1, level-2]
    d = choices[balloon_index % n]
    return max(1, min(MAX_DIFFICULTY, d))


SYMBOL = {
    "addition": "+",
    "subtraction": "−",
    "multiplication": "×",
    "division": "÷",
    "percentage": "% of",
    "fraction": "×",  # we’ll render like "1/3 × 24"
}

class Balloon:
    def __init__(self, *, left:int, operation:str, right:int, answer:str,
                 ttl:int, channel:int, level:int, difficulty:int, balloon_index:int):
        self.id = int(time.time() * 1e6) + random.randint(0, 9999)
        self.left = left
        self.operation = operation
        self.right = right
        self.answer = answer
        self.ttl = int(ttl)  # milliseconds
        self.channel = channel  # 0..3
        self.level = level
        self.difficulty = difficulty
        self.start_time = time.time()
        self.balloon_index = balloon_index  # 0..4 like legacy code

class QuickCalcEngine:
    """
    mechanics_df columns: item_id, a, b, difficulty, speedup, releaseInterval, levelUpHits
    """
    def __init__(self, mechanics_df: pd.DataFrame):
        self.mechanics_df = mechanics_df.sort_values("item_id").reset_index(drop=True)

        # game state
        self.level = 1
        self.correct_count = 0
        self.missed_count = 0
        self.level_hits = 0
        self.balloons: List[Balloon] = []

        # spawn timing
        self.time_since_last_spawn = 0.0
        self.spawn_index = 0  # <- you were missing this

        # logs / flags
        self.balloon_log = []
        self.response_log = []
        self.running = True

    def get_mechanics_row(self, level:int) -> pd.Series:
        idx = max(0, min(level - 1, len(self.mechanics_df) - 1))
        return self.mechanics_df.loc[idx]

    def ttl_from_mechanics(self, speedup: float, balloon_index: int) -> int:
        # Legacy formula: 10000 - speedup + (balloonindex * 500)
        return int(10000 - float(speedup) + (balloon_index * 500))

    # engine.py (replace your spawn_balloon with this)
    def spawn_balloon(self):
        row = self.get_mechanics_row(self.level)

        balloon_index = random.randint(0, 4)
        ttl_ms = self.ttl_from_mechanics(row["speedup"], balloon_index)

        # ← derive difficulty from LEVEL, not from CSV
        d = difficulty_for_level(self.level, balloon_index)

        rule = LEVEL_FUNCTIONS[d]
        op = rule["operation"]

        left, right = sample_operands(rule)
        left, right = postprocess_operands(op, left, right)

        answer = str(compute_numeric(op, left, right))

        new_balloon = Balloon(
            left=left, operation=op, right=right, answer=answer,
            ttl=ttl_ms, channel=random.randint(0, 3),
            level=self.level, difficulty=d, balloon_index=balloon_index
        )
        self.balloons.append(new_balloon)
        self.spawn_index += 1


    def update(self, delta_sec: float):
        if not self.running:
            return

        # spawn with releaseInterval (ms) from current level row
        row = self.get_mechanics_row(self.level)
        release_interval_sec = float(row["releaseInterval"]) / 1000.0

        self.time_since_last_spawn += delta_sec
        if self.time_since_last_spawn >= release_interval_sec:
            self.spawn_balloon()
            self.time_since_last_spawn = 0.0

        # expiry
        to_remove = []
        now = time.time()
        for b in self.balloons:
            age_ms = (now - b.start_time) * 1000.0
            if age_ms >= b.ttl:
                to_remove.append(b)
                self.missed_count += 1
                self.log_balloon_event(b, correct=False, missed=True)

        for b in to_remove:
            if b in self.balloons:
                self.balloons.remove(b)

    def balloon_y_fraction(self, b: Balloon) -> float:
        """0 at top, 1 at bottom? We’ll use 0=bottom, 1=top for clarity in rendering."""
        now = time.time()
        age_ms = (now - b.start_time) * 1000.0
        frac = max(0.0, min(1.0, age_ms / b.ttl))
        return frac  # 0..1 travelled

    def submit_answer(self, user_input:str):
        if not self.balloons:
            return
        try:
            val = str(int(user_input))
        except:
            return
        matches = [b for b in self.balloons if b.answer == val]
        if matches:
            b = matches[0]
            self.correct_count += 1
            self.level_hits += 1
            self.log_balloon_event(b, correct=True, missed=False)
            self.balloons.remove(b)

            # level-up on hits
            row = self.get_mechanics_row(self.level)
            if self.level_hits >= int(row["levelUpHits"]):
                self.level += 1
                self.level_hits = 0
        else:
            self.response_log.append({"timestamp": time.time(), "user_input": user_input, "correct": False})

    def log_balloon_event(self, b:Balloon, correct:bool, missed:bool):
        now = time.time()
        reaction_ms = (now - b.start_time) * 1000.0
        self.balloon_log.append({
            "balloon_id": b.id,
            "level": b.level,
            "difficulty": b.difficulty,
            "operation": b.operation,
            "left": b.left,
            "right": b.right,
            "answer": b.answer,
            "ttl_ms": b.ttl,
            "reaction_ms": reaction_ms if correct else None,
            "correct": correct,
            "missed": missed,
            "timestamp": now,
        })

        # --------------------------------------------------------
    # END SESSION / LOGGING
    # --------------------------------------------------------
    def end_session(self):
        """Save logs if available, but safe to call even if logging is disabled."""
        import pandas as pd, time
        ts = int(time.time())
        try:
            if hasattr(self, "balloon_log") and self.balloon_log:
                pd.DataFrame(self.balloon_log).to_csv(f"quickcalc_balloon_log_{ts}.csv", index=False)
            if hasattr(self, "response_log") and self.response_log:
                pd.DataFrame(self.response_log).to_csv(f"quickcalc_response_log_{ts}.csv", index=False)
            print(f"Session ended. Logs saved with timestamp {ts}.")
        except Exception as e:
            print(f"(Warning) Could not save logs: {e}")

# You already have LEVEL_FUNCTIONS elsewhere; ensure it's accessible.
