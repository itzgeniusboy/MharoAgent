import os, sys
sys.path.insert(0, "src")
root = "src/mharo"
junk_markers = ["ctx7", "serena", "oraios", "usestrix", "############", "VIM-MODAL", "phir se", "Editor:surrogate", "crunch"]
emoji_ok = {"✅", "🚀", "🎯"}  # comments me chhote emoji fine
issues = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, root)
        try:
            txt = open(p, encoding="utf-8").read()
        except Exception as e:
            issues.append(f"{rel}: READ FAIL {e}")
            continue
        n = txt.count("\n") + 1
        # garbage markers
        bad = [m for m in junk_markers if m in txt]
        # compile
        try:
            compile(txt, p, "exec")
            cstat = "OK"
        except SyntaxError as e:
            cstat = f"SYNTAX {e.lineno}"
        # import (module-level)
        mod = "mharo." + os.path.splitext(rel)[0].replace(os.sep, ".")
        try:
            __import__(mod)
            istat = "OK"
        except Exception as e:
            istat = f"IMPORT-FAIL {type(e).__name__}"
        flag = " ".join(bad) or cstat if cstat != "OK" else (" ".join(bad) or "")
        if bad or cstat != "OK" or istat != "OK":
            issues.append(f"{rel}: {n}L COMPILE={cstat} IMPORT={istat} {'JUNK='+','.join(bad) if bad else ''}")
        else:
            print(f"  ok  {rel:55s} {n:4d}L")
print("\n=== ISSUES ===")
for i in issues:
    print(" !!", i)
print("\nTOTAL issues:", len(issues))
