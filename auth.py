from pathlib import Path
import re


__TOKEN2USER: dict = {}


def load_tokens(tokens_path: Path) -> None:
    with open(tokens_path) as f:
        text = f.read()

    global __TOKEN2USER
    __TOKEN2USER = dict(re.findall(r'(\S+): (\S+)', text))


def get_user(token: str) -> str | None:
    return __TOKEN2USER.get(token, None)
