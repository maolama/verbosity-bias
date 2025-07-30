from codes import Environment
from codes.ResponseGeneration import ResponseGeneration
from codes.Evaluator import Evaluator

from codes.Environment import get_config

from time import perf_counter
from contextlib import contextmanager
import os


@contextmanager
def elapsed_time(label="Elapsed"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"{label}: {(end - start) * 1000:.2f} ms")


# create main function
if __name__ == '__main__':
    question_path = "./data/Questions.txt"
    print(Environment.init())

    config = get_config()
    print(config)

    # Translation.translation_route()

    # Response generation
    input_dir = "data/questions"
    answer_dir = "data/responses"
    results_dir = "data/results"
    gen = ResponseGeneration(config=config)
    # with elapsed_time("My code block"):
    #     gen.batch_generate_response(input_dir,
    #                                 answer_dir,
    #                                 model_name="claude3-7",
    #                                 dir_limit=1000,
    #                                 file_limit=1
    #                                 )

    eval_gemma = Evaluator(config=config, model="anthropic/claude-3.7-sonnet")
    with elapsed_time("My code block"):
        eval_gemma.batch_experiments(input_dir,
                                     answer_dir + "/claude3-7",
                                     results_dir,
                                     model_name="claude3-7",
                                     dir_limit=1000,
                                     file_limit=2
                                     )
