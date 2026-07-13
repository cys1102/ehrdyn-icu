from __future__ import annotations

import argparse
import shutil
import stat
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CliArgs(argparse.Namespace):
    source: Path = Path()


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Install the dependency-free KDD2027 package without a build cache.")
    _ = parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args(namespace=CliArgs())


def install(source: Path) -> Path:
    package_source = source / "kdd2027_benchmark"
    purelib = Path(sysconfig.get_paths()["purelib"])
    package_target = purelib / package_source.name
    if package_target.exists():
        raise FileExistsError(f"Package target already exists: {package_target.name}")
    _ = shutil.copytree(package_source, package_target)
    script = f"#!{sys.executable}\nfrom kdd2027_benchmark.cli import main\nraise SystemExit(main())\n"
    launchers = [Path(sys.executable).parent / name for name in ("ehrdyn-icu", "kdd2027")]
    for launcher in launchers:
        _ = launcher.write_text(script, encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launchers[0]


def main() -> int:
    _ = install(parse_args().source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
