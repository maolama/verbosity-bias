import os
import json
import textwrap
from codes.utils import log
from itertools import islice
from tqdm import tqdm

colors = ["cyan", "magenta", "yellow", "red"]

# Set the parent directory containing the 3 subdirectories
parent_dir = '../../data/results/claude3-7'  # <-- Replace with your actual path

# List subdirectories (assumes only 3 and in proper order)
subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
subdirs.sort()  # Sort for consistent order (optional)


# Assume filenames are output1.json to output21.json


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


flag = True
flag2 = False
while flag:
    for ind, subdir in enumerate(subdirs):
        log(f"{ind}. {subdir}", "blue")
    while not flag2:
        try:
            log("Which directory would you like to evaluate?" , color="green" , end="")
            dir_ind = int(input())
            flag2 = True
        except Exception:
            print("try again")
    flag2 = False
    subdir = subdirs[dir_ind]

    print(subdirs[dir_ind])
    files = [f for f in os.listdir(subdir) if f.endswith('.json')]
    print(files)
    flag3 = True
    for batch_files in batched(files, 2):
        if not flag3:
            break
        for ind, filename in enumerate(batch_files):
            file = os.path.join(subdir, filename)
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(data.keys())
            color = colors[ind]
            question = data.get('translated_question', '')
            short_answer = data.get('short_correct', '')
            long_res = data.get('long_restricted', '')
            long_unres = data.get('long_unrestricted', '')
            short_incorrect = data.get('short_incorrect', '')
            short_error_explanation = data.get('short_error_explanation', '')
            long_incorrect = data.get('long_incorrect', '')
            long_error_explanation = data.get('long_error_explanation', '')

            exps = data.get('experiments', {})
            exp0 = exps['0']
            exp1 = exps['1']
            exp2 = exps['2']
            exp3 = exps['3']

            log("question: ", color=color)
            log(textwrap.fill(question, width=140, replace_whitespace=False), color=color)
            log("\n============================================== EXP 0 ==============================================", color=color)

            log("short_answer:", color=color)
            log(textwrap.fill(short_answer, width=140, replace_whitespace=False), color=color)
            log("\nlong_answer:", color=color)
            log(textwrap.fill(long_unres + "\n", width=140, replace_whitespace=False), color=color)
            # log(short_answer, color=color)
            log("********* direct exp *********\n", end="", color=color, on_color="on_grey")
            log(textwrap.fill(exp0['direct']['response'], width=140, replace_whitespace=False) + "\n", color=color,
                on_color="on_grey")
            log("********* reverse exp *********", color=color)
            log(textwrap.fill(exp0['reverse']['response'], width=140, replace_whitespace=False), color=color)

            log("\n============================================== EXP 1 ==============================================", color=color)
            log("short_answer:", color=color)
            log(textwrap.fill(short_answer, width=140, replace_whitespace=False), color=color)
            log("\nlong_restricted:", color=color)
            log(textwrap.fill(long_res, width=140, replace_whitespace=False), color=color)
            # log(short_answer, color=color)
            log("********* direct exp *********\n", end="", color=color, on_color="on_grey")
            log(textwrap.fill(exp1['direct']['response'], width=140, replace_whitespace=False) + "\n", color=color,
                on_color="on_grey")
            log("\n********* reverse exp *********", color=color)
            log(textwrap.fill(exp1['reverse']['response'], width=140, replace_whitespace=False), color=color)

            log("\n============================================== EXP 2 ==============================================", color=color)
            log("short_answer:", color=color)
            log(textwrap.fill(short_answer, width=140, replace_whitespace=False), color=color)
            log("\nlong_answer:", color=color)
            log(textwrap.fill(long_unres, width=140, replace_whitespace=False), color=color)
            # log(short_answer, color=color)
            log("********* direct exp *********\n", end="", color=color, on_color="on_grey")
            log(textwrap.fill(exp2['direct']['response'], width=140, replace_whitespace=False) + "\n", color=color,
                on_color="on_grey")
            log("\n********* reverse exp *********", color=color)
            log(textwrap.fill(exp2['reverse']['response'], width=140, replace_whitespace=False), color=color)

            log("\n============================================== EXP 3 ==============================================", color=color)
            log("short_answer:", color=color)
            log(textwrap.fill(short_answer, width=140, replace_whitespace=False), color=color)
            log("\nlong_incorrect:", color=color)
            log(textwrap.fill(long_incorrect, width=140, replace_whitespace=False), color=color)
            log("\nreason:", color=color)
            log(textwrap.fill(long_error_explanation, width=140, replace_whitespace=False), color=color)
            # log(short_answer, color=color)
            log("********* direct exp *********\n", end="", color=color, on_color="on_grey")
            log(textwrap.fill(exp3['direct']['response'], width=140, replace_whitespace=False) + "\n", color=color,
                on_color="on_grey")
            log("********* reverse exp *********", color=color)
            log(textwrap.fill(exp3['reverse']['response'], width=140, replace_whitespace=False), color=color)
        log("next or exit?" , color="green")
        q = input()
        if q.lower().strip() == 'exit':
            flag3 = False