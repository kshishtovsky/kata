"""Line coverage for kata's own sources, with never-executed files counted as 0%.

gcov only emits a .gcda for a translation unit that ran at least one line, so a
file that no test ever touches is simply ABSENT from its output. Summing only
what gcov reports therefore measures "coverage of the files we happen to
execute", not coverage of the project: src/model.cpp (904 lines, the largest
file) was missing from the report entirely for months and the headline figure
was 21 points too high.

So this script enumerates the sources itself and charges 0% to any file with no
gcda. That makes the number fall when a file is first executed instead of
jumping, which is what actually happened when the routing test was added.
"""
import os
import re
import sys

# coverage.sh runs this from inside build-cov/, so paths must be anchored to the
# project root or the file list comes back empty and every figure is 0%.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
HEADER_DIRS = (os.path.join(ROOT, "include/kata"),)


def source_files():
    files = []
    for d in (SRC_DIR,) + HEADER_DIRS:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith((".cpp", ".h")):
                files.append(os.path.relpath(os.path.join(d, name), ROOT))
    return files


def count_lines(path):
    """Non-blank, non-comment lines: what gcov would consider executable-ish.

    Only used for files gcov never reported, so an approximation is enough to
    weight them; reporting 0% of the real line count would be strictly worse.
    """
    n = 0
    in_block = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if in_block:
                if "*/" in s:
                    in_block = False
                    s = s.split("*/", 1)[1].strip()
                else:
                    continue
            if s.startswith("/*"):
                if "*/" not in s:
                    in_block = True
                continue
            if not s or s.startswith("//"):
                continue
            n += 1
    return n


def main():
    cur = None
    seen = {}
    for line in sys.stdin:
        m = re.match(r"File '(.*)'", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"Lines executed:\s*([\d.]+)% of (\d+)", line)
        if m and cur:
            # relpath, not split("/kata/"): the project path contains "kata"
            # twice (…/kata/include/kata/x.h), so splitting on the last
            # occurrence left "x.h" and every header was reported as
            # never-executed.
            rel = os.path.relpath(cur, ROOT) if cur.startswith(ROOT) else cur
            seen[rel] = (float(m.group(1)), int(m.group(2)))
            # gcov also prints a trailing aggregate "Lines executed" with no
            # File line in front of it; without this reset it is charged to the
            # last file.
            cur = None

    rows = []
    for path in source_files():
        if path in seen:
            rows.append((path, seen[path][0], seen[path][1]))
        else:
            rows.append((path, 0.0, count_lines(os.path.join(ROOT, path))))

    # The headline counts .cpp only. A header is either pure declarations (no
    # executable line, so 0% is meaningless) or inline code that gcov reports
    # twice -- once as the header and once inlined into every .cpp that uses it
    # -- so mixing it into the total double-counts. Header numbers are printed
    # below the line as information.
    cpp = [r for r in rows if r[0].endswith(".cpp")]
    hdr = [r for r in rows if not r[0].endswith(".cpp")]
    total_lines = max(1, sum(n for _, _, n in cpp))
    tot = sum(p * n for _, p, n in cpp) / total_lines
    for f, p, n in sorted(cpp):
        flag = "" if p > 0 or n == 0 else "  <- never executed"
        print(f"{f:28s} {p:6.2f}%  of {n:4d} lines{flag}")
    print(f"TOTAL (src/*.cpp)             {tot:6.2f}%  of {total_lines} lines")
    print()
    print("headers (informational, inline code is counted in its .cpp users):")
    for f, p, n in sorted(hdr):
        print(f"  {f:26s} {p:6.2f}%  of {n:4d} lines")


if __name__ == "__main__":
    main()