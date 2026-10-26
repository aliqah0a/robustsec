"""Write a self-contained manuscript with every \\res{key} replaced by its value.

Usage: python scripts/fill_placeholders.py paper/robustsec_softwarex.tex \\
           paper/results_macros.tex -o paper/robustsec_softwarex_filled.tex

Keys without a value are listed and left as \\res{key}, so they still print
as red [X.XX] placeholders. Run it after scripts/paper_numbers.py.
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("tex")
ap.add_argument("macros")
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

values = dict(re.findall(r"\\defres\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", open(a.macros).read()))
tex = open(a.tex).read()
missing = set()


def sub_pm(m):
    k = m.group(1)
    if k in values and f"{k}/sd" in values:
        return f"{values[k]}\\,$\\pm$\\,{values[k + '/sd']}"
    missing.add(k)
    return m.group(0)


def sub(m):
    k = m.group(1)
    if k in values:
        return values[k]
    missing.add(k)
    return m.group(0)


tex = re.sub(r"\\respm\{([^}]*)\}", sub_pm, tex)
tex = re.sub(r"\\res\{([^}]*)\}", sub, tex)
open(a.out, "w").write(tex)
print(f"written {a.out}; {len(missing)} keys still missing")
for k in sorted(missing):
    print("  missing:", k)
