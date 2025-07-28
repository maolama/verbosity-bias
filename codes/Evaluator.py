from openai import OpenAI
from codes.GlobalVars import *
import json
import os
from tqdm import tqdm
from codes.utils import extract_json_from_text, log, pretty_print
from itertools import islice
import re


class Evaluator:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.evaluator_client = OpenAI(
            base_url=config[RG_API_URL],
            api_key=config[RG_API_KEY],
        )
        pass

    def evaluation_util(self,
                        question,
                        answer_a,
                        answer_b,
                        temperature=0.7,
                        evaluation_prompt=EVALUATION_PROMPTS['persian'],
                        log_enable=False):
        prompt1 = evaluation_prompt.format(QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b)
        if log_enable:
            log(pretty_print(prompt1))

        message1 = {
            "role": "user",
            "content": prompt1
        }

        completion1 = self.evaluator_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[message1]
        )
        return completion1
        pass

    @staticmethod
    def log_util(log_enable, raw_response, handle_position_bias, raw_response_rev):
        if log_enable:
            log("=================== Position 1 ===================")
            log(pretty_print(raw_response), color="cyan")
            log("\n")
            if handle_position_bias:
                log("=================== Position 2 ===================")
                log(pretty_print(raw_response_rev), color="yellow")
        pass

    def experiment0(self,
                    question_file,
                    answer_file,
                    handle_position_bias=True,
                    log_enable=False):
        with open(question_file, 'r', encoding='utf-8') as infile:
            question = json.load(infile)

        with open(answer_file, 'r', encoding='utf-8') as infile:
            answers = json.load(infile)

        completion = self.evaluation_util(
            question=question['translated_question'],
            answer_a=answers['short_correct'],
            answer_b=answers['long_unrestricted'],
            log_enable=log_enable
        )
        raw_response = completion.choices[0].message.content

        raw_response_rev = ""
        completion_rev = None
        if handle_position_bias:
            completion_rev = self.evaluation_util(
                question=question['translated_question'],
                answer_a=answers['long_unrestricted'],
                answer_b=answers['short_correct'],
                log_enable=log_enable
            )
            raw_response_rev = completion_rev.choices[0].message.content

        self.log_util(log_enable, raw_response, handle_position_bias, raw_response_rev)
        if handle_position_bias:
            return completion, completion_rev
        else:
            return completion, None
        pass

    def experiment1(self,
                    question_file,
                    answer_file,
                    handle_position_bias=True,
                    log_enable=False):
        with open(question_file, 'r', encoding='utf-8') as infile:
            question = json.load(infile)

        with open(answer_file, 'r', encoding='utf-8') as infile:
            answers = json.load(infile)

        completion = self.evaluation_util(
            question=question['translated_question'],
            answer_a=answers['short_correct'],
            answer_b=answers['long_restricted'],
            log_enable=log_enable
        )
        raw_response = completion.choices[0].message.content

        raw_response_rev = ""
        completion_rev = None
        if handle_position_bias:
            completion_rev = self.evaluation_util(
                question=question['translated_question'],
                answer_a=answers['long_restricted'],
                answer_b=answers['short_correct'],
                log_enable=log_enable
            )
            raw_response_rev = completion_rev.choices[0].message.content

        self.log_util(log_enable, raw_response, handle_position_bias, raw_response_rev)

        if handle_position_bias:
            return completion, completion_rev
        else:
            return completion, None
        pass

    def experiment2(self,
                    question_file,
                    answer_file,
                    handle_position_bias=True,
                    log_enable=False):
        with open(question_file, 'r', encoding='utf-8') as infile:
            question = json.load(infile)

        with open(answer_file, 'r', encoding='utf-8') as infile:
            answers = json.load(infile)

        completion = self.evaluation_util(
            question=question['translated_question'],
            answer_a=answers['short_correct'],
            answer_b=answers['long_unrestricted'],
            evaluation_prompt=EVALUATION_PROMPTS['english'],
            log_enable=log_enable
        )
        raw_response = completion.choices[0].message.content

        raw_response_rev = ""
        completion_rev = None
        if handle_position_bias:
            completion_rev = self.evaluation_util(
                question=question['translated_question'],
                answer_a=answers['long_unrestricted'],
                answer_b=answers['short_correct'],
                evaluation_prompt=EVALUATION_PROMPTS['english'],
                log_enable=log_enable
            )
            raw_response_rev = completion_rev.choices[0].message.content

        self.log_util(log_enable, raw_response, handle_position_bias, raw_response_rev)

        if handle_position_bias:
            return completion, completion_rev
        else:
            return completion, None
        pass

    def experiment3(self,
                    question_file,
                    answer_file,
                    handle_position_bias=True,
                    log_enable=False):
        with open(question_file, 'r', encoding='utf-8') as infile:
            question = json.load(infile)

        with open(answer_file, 'r', encoding='utf-8') as infile:
            answers = json.load(infile)

        completion = self.evaluation_util(
            question=question['translated_question'],
            answer_a=answers['short_correct'],
            answer_b=answers['long_incorrect'],
            log_enable=log_enable
        )
        raw_response = completion.choices[0].message.content

        raw_response_rev = ""
        completion_rev = None
        if handle_position_bias:
            completion_rev = self.evaluation_util(
                question=question['translated_question'],
                answer_a=answers['long_incorrect'],
                answer_b=answers['short_correct'],
                log_enable=log_enable
            )
            raw_response_rev = completion_rev.choices[0].message.content

        self.log_util(log_enable, raw_response, handle_position_bias, raw_response_rev)

        if handle_position_bias:
            return completion, completion_rev
        else:
            return completion, None
        pass

    def batch_experiments(self,
                          question_bsae_dir,
                          answer_base_dir,
                          output_base_dir,
                          model_name,
                          temperature=0.7,
                          dir_limit=1000,
                          file_limit=1000):
        all_filenames = [f for f in os.listdir(question_bsae_dir)]
        counter = 0
        for ind, file_name in enumerate(all_filenames):
            if counter >= dir_limit:
                break
            self.batch_experiments_util(
                question_dir=f"{question_bsae_dir}/{file_name}/",
                answer_dir=f"{answer_base_dir}/{file_name}/",
                output_dir=f"{output_base_dir}/{model_name}/{file_name}/",
                question_type=file_name,
                temperature=temperature,
                question_limit=file_limit
            )
            counter += 1
        pass

    def batch_experiments_util(self,
                               question_dir,
                               answer_dir,
                               output_dir,
                               question_type,
                               temperature=0.7,
                               question_limit=1000):
        print(f"Start processing {question_type}")

        # Ensure responses directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get all filenames ending with .json
        all_filenames = [f for f in os.listdir(question_dir) if f.endswith('.json')]
        counter = 0
        for ind, file in tqdm(enumerate(all_filenames)):
            if counter >= question_limit:
                break

            question_file = os.path.join(question_dir, file)
            answer_file = os.path.join(answer_dir, file)
            output_path = os.path.join(output_dir, file)
            print(output_path)

            if os.path.exists(output_path):
                print(f"{output_path} already exists")
                continue

            results = self.experiments(question_file,
                                       answer_file,
                                       handle_position_bias=True)
            self.save_result(results, output_path)
            counter += 1
        print(f"Finished processing {question_type}")
        print(50 * "-")

    def experiments(self,
                    question_file,
                    answer_file,
                    handle_position_bias=True,
                    log_enable=True):
        log("========== EXPERIMENT 0 =========", color="green")
        exp0, exp0_rev = self.experiment0(question_file,
                                          answer_file,
                                          handle_position_bias,
                                          log_enable)

        log("========== EXPERIMENT 1 =========", color="green")
        exp1, exp1_rev = self.experiment1(question_file,
                                          answer_file,
                                          handle_position_bias,
                                          log_enable)

        log("========== EXPERIMENT 2 =========", color="green")
        exp2, exp2_rev = self.experiment2(question_file,
                                          answer_file,
                                          handle_position_bias,
                                          log_enable)

        log("========== EXPERIMENT 3 =========", color="green")
        exp3, exp3_rev = self.experiment3(question_file,
                                          answer_file,
                                          handle_position_bias,
                                          log_enable)

        log("========== END EXPERIMENTS=========", color="green")

        result = self.handle_results(question_file, answer_file,
                                     exp0, exp0_rev,
                                     exp1, exp1_rev,
                                     exp2, exp2_rev,
                                     exp3, exp3_rev)
        return result

        # self.save_result(result, save_dir, question_name)
        pass

    @staticmethod
    def save_result(result, save_dir):
        with open(save_dir, "w", encoding='utf-8') as outfile:
            json.dump(result, outfile, ensure_ascii=False, indent=4)
        pass

    def handle_results(self, question_file, answer_file, exp0, exp0_rev, exp1, exp1_rev, exp2, exp2_rev, exp3,
                       exp3_rev):
        with open(question_file, 'r', encoding='utf-8') as infile:
            question = json.load(infile)

        with open(answer_file, 'r', encoding='utf-8') as infile:
            answers = json.load(infile)

        result = {**question, **answers, "experiments": {}}
        result = self.handle_results_util(result, exp0, exp0_rev, "0")
        result = self.handle_results_util(result, exp1, exp1_rev, "1")
        result = self.handle_results_util(result, exp2, exp2_rev, "2")
        result = self.handle_results_util(result, exp3, exp3_rev, "3")

        return result
        pass

    def handle_results_util(self, result, exp, exp_rev, id):
        result["experiments"][id] = {
            "direct": {
                "response": exp.choices[0].message.content,
                "extracted_answer": self.extract_answer(exp.choices[0].message.content)
            }
        }
        if exp_rev is not None:
            result["experiments"][id]["reverse"] = {
                "response": exp_rev.choices[0].message.content,
                "extracted_answer": self.extract_answer(exp_rev.choices[0].message.content)
            }
        return result
        pass

    @staticmethod
    def extract_answer(text):

        matches = re.findall(r'\[\[([A-C])\]\]', text)

        if matches:
            last = matches[-1]
            # print(last)  # خروجی: 'C'
            # print(f'[[{last}]]')  # خروجی: '[[C]]'
            return last
        return None
        pass
