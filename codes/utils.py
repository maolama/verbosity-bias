from termcolor import *
import os
from dotenv import load_dotenv
import re
import json
import textwrap


def log(prompt, color="white", on_color=None, end="\n"):
    cprint(prompt, color=color, on_color=on_color, force_color=True , end=end)


def log_error(prompt, color="red"):
    log(prompt, color)


def extract_json_from_text(text):
    # This regex attempts to find a JSON object or array
    # It looks for the first '{' or '[' and then tries to match
    # until the corresponding closing '}' or ']'
    # This is a basic approach and might fail for complex or malformed JSON
    match = re.search(r'({.*}|\[.*\])', text, re.DOTALL)
    if match:
        json_part = match.group(0)
        try:
            return json_part
        except json.JSONDecodeError:
            # If the initial regex capture isn't perfect JSON,
            # you might need more sophisticated parsing or error handling.
            return None
    return None


def pretty_print(s, width=140):
    return textwrap.fill(s, width=width, replace_whitespace=False)
    pass
