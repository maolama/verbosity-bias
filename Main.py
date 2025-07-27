from codes import Environment
from codes.ResponseGeneration import ResponseGeneration

from codes.Environment import get_config
import os

# create main function
if __name__ == '__main__':
    question_path = "./data/Questions.txt"
    print(Environment.init())
    #### init(question_path)
    # Translation.translation_route()
    config = get_config()
    print(config)
    gen = ResponseGeneration(config=config)
    print(gen)

    input_dir = "data/questions"
    output_dir = "data/output"

    gen.batch_generate_response(input_dir, output_dir, model_name="gemma3")
    # all_filenames = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    # gen.batch_generate_response(
    #     input_dir="data/questions/Reasoning/",
    #     output_dir="data/results/Reasoning/Reasoning_gemma3/",
    #     question_type="استدلال"
    # )
