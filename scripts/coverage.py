import re, sys
cur = None
rows = []
for line in sys.stdin:
    m = re.match(r"File '(.*)'", line)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"Lines executed:\s*([\d.]+)% of (\d+)", line)
    if m and cur and cur.startswith("/home"):
        rows.append((cur.split("/kata/")[-1], float(m.group(1)), int(m.group(2))))
        # gcov also prints a trailing aggregate "Lines executed" with no File
        # line in front of it; without this reset it is charged to the last file.
        cur = None
tot = sum(p * n for _, p, n in rows) / max(1, sum(n for _, _, n in rows))
for f, p, n in sorted(rows):
    print(f"{f:28s} {p:6.2f}%  of {n:4d} lines")
print(f"TOTAL (kata sources)          {tot:6.2f}%")
