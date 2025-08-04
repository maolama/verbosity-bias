import os
import json
import textwrap
from codes.utils import log

colors = ["cyan", "magenta", "green", "yellow"]

# Set the parent directory containing the 3 subdirectories
parent_dir = '../../data/responses/gemma3'  # <-- Replace with your actual path

# List subdirectories (assumes only 3 and in proper order)
subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
subdirs.sort()  # Sort for consistent order (optional)

# Assume filenames are output1.json to output21.json
for subdir in subdirs:
    for i in range(1, 4):
        filename = f'question_{i:02d}.json'
        print(f"\n{'=' * 40} {filename} {'=' * 40}")

        file_path = os.path.join(subdir, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        short_answer = data.get('short_correct', '')
        long_answer = data.get('long_restricted', '')
        long_unres = data.get('long_unrestricted', '')
        short_incorrect = data.get('short_incorrect', '')
        short_error_explanation = data.get('short_error_explanation', '')
        long_incorrect = data.get('long_incorrect', '')
        long_error_explanation = data.get('long_error_explanation', '')

        print(f"\nFrom: {subdir}")
        print("short_answer:")
        print(log(textwrap.fill(short_answer, width=140, replace_whitespace=False), color=colors[i - 1]))
        print("\nlong_answer:")
        print(log(textwrap.fill(long_answer, width=140, replace_whitespace=False), color=colors[i - 1]))
        print("\nlong_unres:")
        print(log(textwrap.fill(long_unres, width=140, replace_whitespace=False), color=colors[i - 1]))
        print("\nshort_incorrect:")
        print(log(textwrap.fill(short_incorrect, width=140, replace_whitespace=False), color=colors[i - 1]))
        print("\nshort_error_explanation:")
        print(log(textwrap.fill(short_error_explanation, width=140, replace_whitespace=False), color=colors[i - 1]))
        print("\nlong_incorrect:")
        print(log(textwrap.fill(long_incorrect, width=140, replace_whitespace=False), color=colors[i - 1]))
        print("\nlong_error_explanation:")
        print(log(textwrap.fill(long_error_explanation, width=140, replace_whitespace=False), color=colors[i - 1]))
