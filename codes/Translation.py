import os
import google.generativeai as genai
from codes.GlobalVars import *
from codes.Environment import get_config

import google.generativeai as genai
from tqdm import tqdm
import json
import time


def translation_route():
    CONFIG = get_config()
    if CONFIG[TR_PR] == "GOOGLE":
        print("google")
        genai.configure(api_key=CONFIG[TR_API_KEY])
        model = genai.GenerativeModel(CONFIG[TR_MODEL])
        print(model)
        google_path(model, CONFIG)
        verify_translation_is_done()
    else:
        print(CONFIG[TR_API_URL])
        print(CONFIG[TR_API_KEY])


def verify_translation_is_done():
    errors =[]
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through each file in the folder
            for file_name in tqdm(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                print(file_path)
                if file_name.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Check if the 'translated_question' key exists
                    if 'translated_question' not in data:
                        errors.append(file_path)

    if len(errors) > 0:
        for error in errors:
            print(error)
    else:
        print("Everything is done")

    pass


def google_path(model, config):
    def translate(q):
        prompt = (
            "You are a translation assistant.\n"
            "Translate the following English text into Persian (Farsi)."
            "Please use simple and natural language (but formal grammer) that is easy to understand. "
            "Avoid using overly formal or literary expressions. The translation should feel fluent and clear for everyday readers. "
            "Consider the structure of Persian language about position of nouns, verbs, adjectives. "
            "Don't explain, just translate. No Finglish.\n\n"
            f"The text is: '{q}'"
        )

        response = model.generate_content(prompt)
        print(response.text)
        return response.text

    counter = 0
    xx = 0
    # Iterate through each folder in the data directory
    for folder_name in os.listdir(data_path):
        print(folder_name)
        folder_path = os.path.join(data_path, folder_name)
        xx = 0
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through each file in the folder
            for file_name in tqdm(os.listdir(folder_path)):
                # if xx > 2:
                #     break
                file_path = os.path.join(folder_path, file_name)
                print(file_path)
                # Check if it's a JSON file
                if file_name.endswith('.json'):
                    try:
                        # Open and load the JSON data
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Check if the 'translated_question' key exists
                        if 'translated_question' in data:
                            xx += 1
                            continue
                        # Check if the 'original_question' key exists
                        if 'original_question' in data:
                            original_question = data['original_question']
                            # Translate the question
                            translated_question = translate(original_question)
                            # Add the translated question to the data
                            data['translated_question'] = translated_question

                            # Save the updated data back to the JSON file
                            with open(file_path, 'w', encoding='utf-8') as outfile:
                                json.dump(data, outfile, indent=2,
                                          ensure_ascii=False)  # ensure_ascii=False to handle Persian characters
                            counter += 1
                            xx += 1
                            if counter % 8 == 0:
                                time.sleep(60)

                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    pass
