import logging
from typing import Dict


def read(fpath: str):
    with open(fpath, "r") as f:
        content = f.read()
    return content


def interpolate(content: str, mappings: Dict[str, str]) -> str:
    for key, value in mappings.items():
        content = content.replace(key, value)
    return content


def export(content: str, fpath: str) -> None:
    with open(fpath, "w") as f:
        f.write(content)
    logging.info(f"File exported under: {fpath}")
