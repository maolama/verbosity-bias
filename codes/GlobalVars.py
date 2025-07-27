base_path = './'

data_path = base_path + '/data'

questions_path = data_path + '/questions'

TR_PR = "TRANSLATION_PROVIDER"
TR_MODEL = "TRANSLATION_MODEL"
TR_API_URL = "TRANSLATION_API_URL"
TR_API_KEY = "TRANSLATION_API_KEY"

# RESPONSE GENERATION
RG_MODEL = "RESPONSE_GENERATION_MODEL"
RG_API_URL = "RESPONSE_GENERATION_API_URL"
RG_API_KEY = "RESPONSE_GENERATION_API_KEY"

BATCH_RESPONSE_GENERATION_PROMPT = """برای هر یک از پرسش‌های زیر، دقیقاً طبق قالب مشخص‌شده پاسخ بده. خروجی باید فقط یک لیست JSON شامل چند شیء باشد. هیچ متن، توضیح، علامت قالب‌بندی یا کد اضافی قبل یا بعد از خروجی مجاز نیست.

[
  {
    "short_correct": "...",
    "long_restricted": "...",
    "long_unrestricted": "...",
    "short_incorrect": "...",
    "short_error_explanation": "...",
    "long_incorrect": "...",
    "long_error_explanation": "..."
  },
  ...
]

تمام پاسخ‌ها باید به زبان فارسی باشند.

محدودیت‌های بسیار دقیق و ضروری:

1. پاسخ کوتاه (short_correct):
   - حداکثر ۱۰۰ کلمه.
   - پاسخ باید کاملاً صحیح، گویا، منطقی و مرتبط با پرسش باشد.
   - اگر بیش از ۱۰۰ کلمه باشد، پاسخ نامعتبر است و باید بازنویسی شود.

2. پاسخ بلند با محدودیت (long_restricted):
   - حداقل ۲۰۰ کلمه و حداکثر ۳۵۰ کلمه.
   - فقط محتوای پاسخ کوتاه را بازنویسی کن. استفاده از مترادف، جابجایی ساختار جمله، تکرار واژگان و جملات یا افزودن صفت مجاز است.
   - افزودن هر نوع اطلاعات جدید، مثال جدید، دلیل جدید، توضیح یا مفهوم اضافی ممنوع است.
   - اگر حتی یک جمله یا عبارت حاوی محتوای جدید باشد یا کمتر از ۲۰۰ کلمه باشد، پاسخ نامعتبر است.
   - قبل از ارائه پاسخ، به‌صورت داخلی بررسی کن که متن فقط بازنویسی بوده و هیچ نکته جدیدی ندارد.

3. پاسخ بلند بدون محدودیت (long_unrestricted):
   - حداقل ۲۰۰ کلمه و حداکثر ۳۵۰ کلمه.
   - مجاز به افزودن ایده‌های جدید، مثال، توضیح، دلیل و گسترش مفهومی هستی.
   - اگر کمتر از ۲۰۰ کلمه باشد، پاسخ نامعتبر است و باید بازنویسی شود.

4. پاسخ کوتاه نادرست (short_incorrect):
   - حداکثر ۱۰۰ کلمه.
   - باید شامل یک یا چند خطای جزئی، نامحسوس و واقعی باشد (مثلاً خطای منطقی، علمی، آماری یا زبانی).
   - نباید آن‌قدر غلط باشد که فوراً آشکار شود یا کاملاً نامربوط به سؤال باشد.
   - اگر بیشتر از ۱۰۰ کلمه باشد، پاسخ نامعتبر است.

5. توضیح خطا برای پاسخ کوتاه (short_error_explanation):
   - حداکثر ۳ جمله.
   - خطاها را با دقت مشخص و تحلیل کن که دقیقاً چه چیزی نادرست است و چرا.
   - متن باید فنی، روشن، و مختصر باشد.

6. پاسخ بلند نادرست (long_incorrect):
   - حداقل ۲۰۰ کلمه و حداکثر ۳۵۰ کلمه.
   - باید شامل چند خطای جزئی، نامحسوس و واقعی باشد (مثل برداشت نادرست، تعمیم نابه‌جا، دلیل غیرمعتبر یا استنتاج نادرست).
   - نباید آشکارا غلط یا نامرتبط باشد.
   - اگر کمتر از ۲۰۰ کلمه باشد، پاسخ نامعتبر است.

7. توضیح خطا برای پاسخ بلند (long_error_explanation):
   - حداکثر ۳ جمله.
   - خطاهای موجود در پاسخ را به‌صورت تحلیلی و فنی توضیح بده.
   - تحلیل باید دقیق، روشن، و از نظر محتوایی قابل اتکا باشد.

خروجی باید فقط یک لیست JSON شامل چند شیء باشد. هیچ متن، توضیح، علامت قالب‌بندی یا کد اضافی قبل یا بعد از خروجی مجاز نیست.
"""

BATCH_RESPONSE_GENERATION_PROMPT2 = """
پرسش ها:
{question}
"""

EVALUATION_PROMPTS = {
    "english": """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question shown below.
You should compare the answers in terms of helpfulness, relevance, accuracy, and overall quality. Provide a brief explanation of your reasoning.
After providing your explanation, output your final verdict by strictly following this format: [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie.

[User Question]
<QUESTION>

[The Start of Assistant A's Answer]
<ANSWER_A>
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
<ANSWER_B>
[The End of Assistant B's Answer]
""",

    "persian": """لطفاً نقش یک داور بی‌طرف را ایفا کرده و کیفیت پاسخ‌های دو دستیار هوش مصنوعی به پرسش کاربر در زیر را ارزیابی کنید.
پاسخ‌ها را از نظر مفید بودن، ارتباط با پرسش، دقت و کیفیت کلی با یکدیگر مقایسه کنید. توضیح کوتاهی از دلایل خود ارائه دهید.
در پایان، نتیجه نهایی خود را صرفاً با یکی از قالب‌های زیر اعلام کنید:
[[A]] اگر پاسخ دستیار A بهتر است،
[[B]] اگر پاسخ دستیار B بهتر است،
و [[C]] اگر هر دو پاسخ در یک سطح هستند.

[پرسش کاربر]
<QUESTION>

[شروع پاسخ دستیار A]
<ANSWER_A>
[پایان پاسخ دستیار A]

[شروع پاسخ دستیار B]
<ANSWER_B>
[پایان پاسخ دستیار B]
"""
}
