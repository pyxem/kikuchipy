import re
import sys

from outdated import check_outdated


with open("kikuchipy/release.py") as fid:
    for line in fid:
        if line.startswith("version"):
            VERSION = line.strip().split(" = ")[-1][1:-1]

try:
    is_outdated, latest_version = check_outdated("kikuchipy", VERSION)
except ValueError as e:
    latest_version = re.findall(r"\s([\d.]+)", e.args[0])[1]
    is_outdated = True

print(
    f"Outdated:\t\t{is_outdated}\n"
    f"PyPI version:\t\t{latest_version}\n"
    f"'main' branch version:\t{VERSION}"
)

if is_outdated:
    sys.exit(10)
else:
    sys.exit(20)
