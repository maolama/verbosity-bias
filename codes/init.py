import os

from codes.GlobalVars import *


def create_data_dir():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return True


# create required folders
# prompt: read '/content/Questions.txt' and print its content line by line
def create_questions_dir(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('###'):
                # extract ### part
                title = line.replace('###', '').strip()
                if title:
                    folder_path = data_path + "/" + title
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    print(title)


def count_questions(path):
    import json
    counter = 0
    total = 0
    title = ''
    start = False

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('###'):
                start = True
                if title != '':
                    print(f"{title}: {counter}")
                counter = 0
                title = line.replace('###', '').strip()
                continue
            # check if line is empty
            if line.strip() == '':
                continue
            else:
                if not start:
                    continue
                counter += 1
                total += 1
                qu = {"original_question": line.strip()}
                # save qu in json
                if title:
                    folder_path = data_path + "/" + title
                    file_path = os.path.join(folder_path, f"question_{counter:02}.json")
                    with open(file_path, 'w', encoding='utf-8') as outfile:
                        json.dump(qu, outfile, indent=2)

    if title:
        print(f"{title}: {counter}")
    print(f"Total questions: {total}")


def init(path):
    create_data_dir()
    create_questions_dir(path)
    count_questions(path)
    return True
