from codes import Environment
from codes.ResponseGeneration import ResponseGeneration
from codes.Evaluator import Evaluator

from codes.Environment import get_config
import os

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
    gen.batch_generate_response(input_dir, answer_dir, model_name="gemma3")

    eval_gemma = Evaluator(config=config, model="google/gemma-3-27b-it:free")
    # eval_gemma.experiments("data/questions/Algorithmic/question_09.json",
    #                        "data/responses/gemma3/Algorithmic/question_09.json",
    #                        "question_09.json",
    #                        "res")

    eval_gemma.batch_experiments(input_dir,
                                 answer_dir+"/gemma3",
                                 results_dir,
                                 model_name="gemma3",
                                 dir_limit=1000,
                                 file_limit=1
                                 )
