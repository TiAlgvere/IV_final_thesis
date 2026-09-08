from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a converted Elmer mesh.")
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=ROOT / "mesh" / "elmer",
        help="Directory containing the converted Elmer mesh.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh_dir = args.mesh_dir / "mesh" if (args.mesh_dir / "mesh").exists() else args.mesh_dir
    mesh_file = mesh_dir / "mesh.header"
    names_file = mesh_dir / "mesh.names"
    if not mesh_file.exists():
        print(f"Missing Elmer mesh header: {mesh_file}")
        return 1

    print("mesh.header content:")
    print(mesh_file.read_text(encoding="utf-8", errors="replace"))

    header_lines = [line.strip() for line in mesh_file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    counts_line = header_lines[0] if header_lines else ""
    counts = [int(value) for value in re.findall(r"\d+", counts_line)]
    if len(counts) >= 3:
        print(f"number of nodes: {counts[0]}")
        print(f"number of elements: {counts[1]}")
        print(f"number of boundary elements: {counts[2]}")
    else:
        print("number of nodes: MISSING")
        print("number of elements: MISSING")
        print("number of boundary elements: MISSING")

    body_ids: list[str] = []
    boundary_ids: list[str] = []
    if names_file.exists():
        section = None
        for line in names_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("!"):
                lower = line.lower()
                if "bodies" in lower:
                    section = "body"
                elif "boundaries" in lower:
                    section = "boundary"
                continue
            if not line.startswith("$") or section is None:
                continue
            match = re.match(r"\$\s*(.*?)\s*=\s*(\d+)$", line)
            if not match:
                continue
            name, identifier = match.groups()
            if section == "body":
                body_ids.append(f"{name}={identifier}")
            elif section == "boundary":
                boundary_ids.append(f"{name}={identifier}")

    print(f"body/material IDs present: {', '.join(body_ids) if body_ids else 'MISSING'}")
    print(f"boundary IDs present: {', '.join(boundary_ids) if boundary_ids else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
