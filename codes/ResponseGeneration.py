from openai import OpenAI
from codes.GlobalVars import *


class ResponseGeneration:

    def __init__(self, config):
        self.config = config
        self.response_generator_model = OpenAI(
            base_url=config[RG_API_URL],
            api_key=config[RG_API_KEY],
        )
        pass

    def generate_persian_judgment_prompt_chat(question: str, response_a: str, response_b: str) -> list:

        system_message = {
            "role": "system",
            "content": (
                "لطفاً به‌عنوان یک داور بی‌طرف عمل کنید و کیفیت دو پاسخ متفاوت به یک سؤال یکسان را مقایسه کنید. "
                "پس از ارائه‌ی استدلال خود، تصمیم نهایی را دقیقاً با یکی از قالب‌های زیر بیان کنید: "
                "اگر پاسخ A بهتر است بنویسید [[A]]، اگر پاسخ B بهتر است بنویسید [[B]]، و اگر هر دو پاسخ در یک سطح هستند، بنویسید [[C]]."
            )
        }

        user_message = {
            "role": "user",
            "content": f"""
    [سؤال کاربر]
    {question}

    [شروع پاسخ A]
    {response_a}
    [پایان پاسخ A]

    [شروع پاسخ B]
    {response_b}
    [پایان پاسخ B]
    """.strip()
        }

        return [system_message, user_message]
    def generate_base_response(self):

        pass

    def experiment1_response(self):
        pass

    def experiment2_response(self):
        pass

    def experiment3_response(self):
        pass