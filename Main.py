import json

from codes import Environment
from codes.ResponseGeneration import ResponseGeneration
from codes.Evaluator import Evaluator

from codes.Environment import get_config

from time import perf_counter
from contextlib import contextmanager
import os


@contextmanager
def elapsed_time(label="Elapsed", result_container=None):
    start = perf_counter()
    yield  # Yield control to the block
    end = perf_counter()
    elapsed = (end - start) * 1000  # milliseconds
    print(f"{label}: {elapsed:.2f} ms")
    if result_container is not None:
        result_container["elapsed"] = elapsed


# create main function
if __name__ == '__main__':
    question_path = "./data/Questions.txt"
    print(Environment.init())

    config = get_config()
    print(config)

    # Translation.translation_route()

    # Response generation
    input_dir = "./data/questions"
    answer_dir = "./data/responses"
    results_dir = "./data/evaluation"
    gen = ResponseGeneration(config=config)
    timer = {}
    with elapsed_time("My code block", timer):
        total_input, total_output = gen.batch_generate_response_list(input_bsae_dir=input_dir,
                                                                     output_base_dir=answer_dir,
                                                                     model_name="claude3-7",
                                                                     list_dir=['Algorithmic', 'Math',
                                                                               'Creative-Writing'],
                                                                     temperature=0.7,
                                                                     file_limit=1000
                                                                     )
    # os.makedirs("logs/", exist_ok=True)
    # with open("logs/response_generation.txt", "w", encoding='UTF-8') as f:
    #     Q = str(total_input) + "|" + str(total_output) + "|" + str(timer['elapsed'])
    #     json.dump(Q, f, ensure_ascii=False, indent=4)

    evaluators = {
        'deepseek-r1': 'deepseek/deepseek-r1-0528:free',
        'gemma3': 'google/gemma-3-27b-it:free',
    }
    for name, evaluator_model in evaluators.items():
        print(name, evaluator_model)
        evaluator = Evaluator(config=config, model=evaluator_model)
        timer = {}
        with elapsed_time(name, timer):
            total_input, total_output = evaluator.batch_experiments_list(question_bsae_dir=input_dir,
                                                                         answer_base_dir=answer_dir + "/claude3-7",
                                                                         output_base_dir=results_dir,
                                                                         list_dir=['Algorithmic' ,'Math',
                                                                               'Creative-Writing'],
                                                                         model_name=name,
                                                                         temperature=0.7,
                                                                         file_limit=1000
                                                                         )
        with open(f"logs/data_{name}.txt", "w", encoding='UTF-8') as f:
            Q = str(total_input) + "|" + str(total_output) + "|" + str(timer['elapsed'])
            json.dump(Q, f, ensure_ascii=False, indent=4)
