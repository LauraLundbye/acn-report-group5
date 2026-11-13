
import argparse as ap

def setup_argparser(description: str, subject:bool = True):
    parser = ap.ArgumentParser(description=description)
    if subject:
        parser.add_argument("--subject", "-s", default="001", help="Subject ID (e.g., 001)")
    return parser