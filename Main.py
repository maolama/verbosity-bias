from codes.init import init
from codes import Environment
from codes import Translation
import google.generativeai as genai

from codes.Environment import get_config
from codes.GlobalVars import *


# create main function
if __name__ == '__main__':
    question_path = "./data/Questions.txt"
    print(Environment.init())
    #### init(question_path)
    # Translation.translation_route()
    print("google")
    CONFIG = get_config()
    genai.configure(api_key=CONFIG[TR_API_KEY])
    model = genai.GenerativeModel(CONFIG[TR_MODEL])
    print(model)

    prompt = (
        "How can I open a chrome (or any visual browser) on google colab and then connect to it via my local host?"
    )

    response = model.generate_content(prompt)
    print(response.text)

