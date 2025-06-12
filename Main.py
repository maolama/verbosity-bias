from codes.init import init
from codes import Environment
from codes import Translation


# create main function
if __name__ == '__main__':
    question_path = "./data/Questions.txt"
    print(Environment.init())
    # init(question_path)
    Translation.translation_route()
