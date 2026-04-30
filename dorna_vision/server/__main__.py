import argparse

from .app import main, DEFAULT_PORT


def cli():
    parser = argparse.ArgumentParser(prog="dorna_vision.server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    main(host=args.host, port=args.port, max_workers=args.max_workers)


if __name__ == "__main__":
    cli()
