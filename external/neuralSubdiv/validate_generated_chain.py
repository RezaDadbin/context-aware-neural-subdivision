"""Validate one C++-generated OBJ chain before accepting its random seed."""

from __future__ import print_function

import argparse
import json
from types import SimpleNamespace

from include import tgp
from validate_dataset import validate_chain, validation_settings


def load_chain(paths):
    chain = []
    for path in paths:
        vertices, faces = tgp.readOBJ(path)
        chain.append(SimpleNamespace(V=vertices, F=faces))
    return chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="ordered output_sN.obj files")
    parser.add_argument("--expected-start-faces", type=int, required=True)
    parser.add_argument("--max-narrow-ratio", type=float, default=0.1)
    parser.add_argument("--prefix-tolerance", type=float, default=1e-6)
    parser.add_argument("--min-core-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--min-core-face-area-ratio", type=float, default=1e-10)
    parser.add_argument("--min-core-normal-sum", type=float, default=1e-3)
    parser.add_argument("--allow-boundary", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = validation_settings(
        max_narrow_ratio=args.max_narrow_ratio,
        prefix_tolerance=args.prefix_tolerance,
        min_core_edge_ratio=args.min_core_edge_ratio,
        min_core_face_area_ratio=args.min_core_face_area_ratio,
        min_core_normal_sum=args.min_core_normal_sum,
        allow_boundary=args.allow_boundary,
    )
    chain = load_chain(args.paths)
    result = validate_chain(chain, 0, settings)
    for level, mesh in enumerate(chain):
        expected = args.expected_start_faces * (4 ** level)
        actual = int(mesh.F.size(0))
        if actual != expected:
            result["failures"].append(
                "chain 0 level %d: expected %d faces, found %d" %
                (level, expected, actual))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["failures"]:
        print("; ".join(result["failures"]))
    else:
        print("valid")
    raise SystemExit(1 if result["failures"] else 0)


if __name__ == "__main__":
    main()
