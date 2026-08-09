#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "v2.4.0"
ASSET_NAME = "OpenMVS_Ubuntu_x64.zip"
URL = (
    "https://github.com/cdcseacave/openMVS/releases/download/"
    f"{VERSION}/{ASSET_NAME}"
)
SHA256 = "7104ae1ddd6ca38fbca9e0e4a70b20af59e21e0b497eb7181c864fbf38ca8d00"

REQUIRED = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    def report(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = 100.0 * done / total
        print(f"\rDownloading: {pct:6.2f}%  {done / 1024**2:7.1f}/{total / 1024**2:.1f} MiB",
              end="", flush=True)

    print(f"Downloading official OpenMVS {VERSION}:")
    print(url)
    urllib.request.urlretrieve(url, dst, reporthook=report)
    print()


def find_binary(root: Path, name: str) -> Path | None:
    matches = []
    for path in root.rglob(name):
        if path.is_file():
            matches.append(path)
    if not matches:
        return None

    # Prefer files already under a bin directory.
    matches.sort(key=lambda p: ("bin" not in [part.lower() for part in p.parts], len(p.parts)))
    return matches[0]


def check_binary(path: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [str(path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        # Many CLI programs return non-zero for --help, so successful execution
        # is more important than the return code.
        text = (proc.stdout or "").strip()
        return True, text[:500]
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install official prebuilt OpenMVS Ubuntu x64 binaries."
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=Path("~/OpenMVS-v2.4.0"),
        help="Installation directory.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the release archive again even if cached.",
    )
    args = parser.parse_args()

    if platform.system() != "Linux":
        raise SystemExit(
            f"This installer is for Linux/Ubuntu; detected {platform.system()}."
        )

    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise SystemExit(
            f"The official asset used by this script is x64; detected {machine}."
        )

    install_root = args.install_root.expanduser().resolve()
    cache_dir = Path("~/.cache/openmvs").expanduser()
    archive = cache_dir / ASSET_NAME

    if args.force_download and archive.exists():
        archive.unlink()

    if not archive.exists():
        download(URL, archive)
    else:
        print(f"Using cached archive: {archive}")

    print("Verifying SHA-256...")
    actual = sha256_file(archive)
    if actual.lower() != SHA256.lower():
        archive.unlink(missing_ok=True)
        raise SystemExit(
            "SHA-256 mismatch.\n"
            f"Expected: {SHA256}\n"
            f"Actual:   {actual}\n"
            "The archive was removed. Run the installer again."
        )
    print("SHA-256 OK.")

    extract_root = install_root / "release"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting to: {extract_root}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_root)

    found: dict[str, Path] = {}
    for name in REQUIRED:
        path = find_binary(extract_root, name)
        if path is None:
            raise SystemExit(f"Could not find required OpenMVS binary: {name}")
        path.chmod(path.stat().st_mode | 0o111)
        found[name] = path.resolve()

    # Create one stable directory for the downstream scripts.
    stable_bin = install_root / "bin"
    stable_bin.mkdir(parents=True, exist_ok=True)
    for name, target in found.items():
        link = stable_bin / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    print("\nInstalled binaries:")
    for name in REQUIRED:
        print(f"  {name}: {found[name]}")

    print("\nSmoke test:")
    failed = False
    for name in REQUIRED:
        ok, detail = check_binary(stable_bin / name)
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
        if not ok:
            failed = True
            print(f"    {detail}")

    print("\nOpenMVS bin directory:")
    print(stable_bin)

    print("\nUse it with Sofa50:")
    print(
        "python scripts/prepare_sofa50_openmvs_1920.py "
        f'--openmvs-bin-dir "{stable_bin}"'
    )

    print("\nOptional PATH setup for the current shell:")
    print(f'export PATH="{stable_bin}:$PATH"')

    if failed:
        print(
            "\nThe files were installed, but at least one executable could not start.\n"
            "This normally indicates a missing system shared library. Run:\n"
            f'  ldd "{stable_bin / "DensifyPointCloud"}" | grep "not found"\n'
            "and inspect the missing libraries."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())