#!/usr/bin/env python3
import argparse
from kb import KnowledgeBase

def main():
    parser = argparse.ArgumentParser(description="Local KB indexer")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Fresh build of the vector DB")
    sub.add_parser("update", help="Incremental update (add/change/remove files)")
    sub.add_parser("rebuild", help="Full rebuild (same as build)")

    args = parser.parse_args()
    kb = KnowledgeBase(config_path=args.config)

    if args.cmd == "build" or args.cmd == "rebuild":
        kb.build()
    elif args.cmd == "update":
        kb.update()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
