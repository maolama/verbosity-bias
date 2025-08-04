from openai import OpenAI
from codes.GlobalVars import *
import json
import os
from tqdm import tqdm
from codes.utils import extract_json_from_text
from itertools import islice


class ResponseGeneration:

    def __init__(self, config):
        self.config = config
        self.response_generator_model = OpenAI(
            base_url=config[RG_API_URL],
            api_key=config[RG_API_KEY],
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        pass

    @staticmethod
    def word_count(text):
        dict_str = str(text)
        # Count number of words
        return len(dict_str.split())

    def generate_response(self,
                          question,
                          default_gen1_prompt,
                          default_gen2_prompt,
                          temperature=0.7):
        prompt = default_gen2_prompt.format(question=question)

        message = {
            "role": "user",
            "content": default_gen1_prompt + prompt
        }
        self.total_input_tokens += self.word_count(message)

        completion = self.response_generator_model.chat.completions.create(
            model=self.config[RG_MODEL],
            temperature=temperature,
            messages=[message]
        )
        return completion

        pass

    @staticmethod
    def create_question_string(questions):
        res = ""
        for ind, question in enumerate(questions):
            res += str((ind + 1)) + ". " + question['translated_question'] + "\n"
        return res[:-1]
        pass

    def batch_generate_response(self,
                                input_bsae_dir: str,
                                output_base_dir: str,
                                model_name: str,
                                temperature: float = 0.7,
                                dir_limit: int = 1000,
                                file_limit: int = 1000
                                ):
        all_filenames = [f for f in os.listdir(input_bsae_dir)]
        counter = 0
        for file_name in all_filenames:
            if counter >= dir_limit:
                break
            self.batch_generate_response_util(
                input_dir=f"{input_bsae_dir}/{file_name}/",
                output_dir=f"{output_base_dir}/{model_name}/{file_name}/",
                question_type=file_name,
                temperature=temperature,
                question_limit=file_limit
            )
            counter += 1
        pass

    def batch_generate_response_list(self,
                                input_bsae_dir: str,
                                output_base_dir: str,
                                model_name: str,
                                list_dir: list,
                                temperature: float = 0.7,
                                file_limit: int = 1000
                                ):
        all_filenames = [f for f in os.listdir(input_bsae_dir)]
        counter = 0
        for file_name in all_filenames:
            if file_name in list_dir:
                print(file_name)
                self.batch_generate_response_util(
                    input_dir=f"{input_bsae_dir}/{file_name}/",
                    output_dir=f"{output_base_dir}/{model_name}/{file_name}/",
                    question_type=file_name,
                    temperature=temperature,
                    question_limit=file_limit
                )
            counter += 1
        return self.total_input_tokens , self.total_output_tokens
        pass

    def batch_generate_response_util(self,
                                     input_dir,
                                     output_dir,
                                     question_type,
                                     temperature=0.7,
                                     question_limit=1000
                                     ):
        print(f"Start processing {question_type}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get all filenames ending with .json
        all_filenames = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        counter = 0

        # Helper function to group items in chunks of n
        def batched(iterable, n):
            it = iter(iterable)
            while batch := list(islice(it, n)):
                yield batch

        # Process in batches of 3
        for batch_files in tqdm(batched(all_filenames, 2)):
            if counter >= question_limit:
                break
            output_path = os.path.join(output_dir, batch_files[-1])
            if os.path.exists(output_path):
                print(f"{output_path} already exists")
                counter += 1
                continue

            questions = []
            for filename in batch_files:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                with open(input_path, 'r', encoding='utf-8') as infile:
                    question = json.load(infile)
                    questions.append(question)

            question_string = self.create_question_string(questions)
            completion = self.generate_response(question=question_string,
                                                temperature=temperature,
                                                default_gen1_prompt=BATCH_RESPONSE_GENERATION_PROMPT,
                                                default_gen2_prompt=BATCH_RESPONSE_GENERATION_PROMPT2)
            raw_response = completion.choices[0].message.content
            self.total_output_tokens += len(raw_response.strip().split())
            response = extract_json_from_text(raw_response)
            response = json.loads(response)

            for ind, filename in enumerate(batch_files):
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    json.dump(response[ind], outfile, ensure_ascii=False, indent=4)
            # break
            counter += 1
        self.verify_generation(input_dir, output_dir)
        print(f"Finished processing {question_type}")
        print(50 * "-")

    def verify_generation(self, input_dir, output_dir):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get all filenames ending with .json
        all_filenames = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        not_generated_files = []
        for filename in all_filenames:
            if not os.path.exists(os.path.join(output_dir, filename)):
                not_generated_files.append(filename)
        print(not_generated_files)
