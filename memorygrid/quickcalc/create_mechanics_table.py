# quikcalc_build_mechanics.py
import re
import ast
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_FILE = PROJECT_ROOT / "data/quickcalc/QuikCalc Main.txt"

def parse_js_object(txt):
    """Convert JS-like objects into Python dicts."""
    txt = txt.replace("null", "None")
    txt = txt.replace("true", "True").replace("false", "False")
    txt = re.sub(r"(\w+):", r'"\1":', txt)
    return ast.literal_eval(txt)

# ---------------------------
# 1. Parse levelFunctions[…]
# ---------------------------
def extract_level_functions(text):
    pattern = re.compile(r"levelFunctions\[(\d+)\]\s*=\s*(\{.*?\});", re.S)
    funcs = {}
    for level, obj in pattern.findall(text):
        d = parse_js_object(obj)
        funcs[int(level)] = d
    return funcs

# ---------------------------
# 2. Parse levelAttributes[…]
# ---------------------------
def extract_level_attributes(text):
    pattern = re.compile(r"levelAttributes\[(\d+)\]\s*=\s*(\{.*?\});", re.S)
    attrs = {}
    for level, obj in pattern.findall(text):
        d = parse_js_object(obj)
        attrs[int(level)] = d
    return attrs

# ---------------------------
# 3. Level → difficulty function mapping
# ---------------------------
def build_level_mapping(max_level=30):
    mapping = {}
    mapping[1] = [1,1,1]
    mapping[2] = [2,1,2]

    for lvl in range(3, max_level+1):
        mapping[lvl] = [lvl, lvl-1, lvl-2]
    return mapping

# ---------------------------
# 4. Fill missing attribute levels via inheritance
# ---------------------------
def expand_attributes(attrs):
    expanded = {}
    for lvl in sorted(attrs.keys()):
        expanded[lvl] = attrs[lvl]

    max_lvl = max(attrs.keys())
    for lvl in range(1, max_lvl+1):
        if lvl not in expanded:
            prev = max(k for k in expanded.keys() if k < lvl)
            expanded[lvl] = expanded[prev]

    return expanded

# ---------------------------
# 5. Build mechanics table
# ---------------------------
def build_mechanics_table(funcs, attrs, mapping):
    rows = []
    for level in sorted(mapping.keys()):
        diff_list = mapping[level]

        function_objs = [funcs[d] for d in diff_list if d in funcs]
        operations = sorted(list({f["operation"] for f in function_objs}))
        attr = attrs.get(level, {})

        speedup = attr.get("speedup", 0)
        lifetime_min = 10000 - speedup + (0 * 500)
        lifetime_max = 10000 - speedup + (4 * 500)

        rows.append({
            "level": level,
            "difficulty_functions": diff_list,
            "operations": operations,
            "n_functions": len(function_objs),
            "releaseInterval": attr.get("releaseInterval"),
            "speedup": speedup,
            "bonus": attr.get("bonus"),
            "balloon_lifetime_min": lifetime_min,
            "balloon_lifetime_max": lifetime_max,
            "functions_raw": function_objs
        })
    return pd.DataFrame(rows)

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Load Main.txt
    with open(MAIN_FILE, "r") as f:
        text = f.read()

    funcs = extract_level_functions(text)
    attrs = extract_level_attributes(text)
    mapping = build_level_mapping(max_level=30)
    attrs_full = expand_attributes(attrs)

    df = build_mechanics_table(funcs, attrs_full, mapping)

    # NEW OUTPUT DIRECTORY
    out_dir = Path("quickcalc_interpolation")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "quikcalc_mechanics.csv"
    df.to_csv(out_file, index=False)

    print(df.head())
    print(f"Saved → {out_file}")