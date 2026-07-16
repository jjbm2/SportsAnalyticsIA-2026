from __future__ import annotations

import argparse
import json

from machine_learning.model_registry import MODEL_FILES, ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Administración manual y segura de modelos")
    parser.add_argument("action", choices=["list", "recommend", "promote", "rollback"])
    parser.add_argument("sport", choices=sorted(MODEL_FILES))
    parser.add_argument("--version")
    parser.add_argument("--backup")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    registry = ModelRegistry()

    if args.action == "list":
        result = registry.list_versions(args.sport)
    elif args.action == "recommend":
        if not args.version:
            parser.error("recommend requiere --version")
        result = registry.recommendation(args.sport, args.version)
    elif args.action == "promote":
        if not args.version:
            parser.error("promote requiere --version")
        result = {"backup": str(registry.promote(args.sport, args.version, confirm=args.confirm))}
    else:
        result = {"restored": str(registry.rollback(args.sport, args.backup, confirm=args.confirm))}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
