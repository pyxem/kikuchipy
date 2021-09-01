import re
import sys

from outdated import check_outdated


with open("kikuchipy/release.py") as fid:
    for line in fid:
        if line.startswith("version"):
            branch_version = line.strip().split(" = ")[-1][1:-1]

try:
    make_release, pypi_version = check_outdated("kikuchipy", branch_version)
except ValueError as e:
    pypi_version = re.findall(r"\s([\d.]+)", e.args[0])[1]
    is_outdated = True

print(make_release)
print(pypi_version)
print(branch_version)
