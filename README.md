# Verbosity Bias in LLM Judges for Persian

Do LLMs that act as judges prefer **longer** answers, even when the extra length adds no information, or the long answer is wrong? This project measures that *verbosity bias* in a Persian, low-resource setting.

*Course research project for the graduate Large Language Models course at TEIAS (Spring 2025). Main author: Mohammad Ali Olama; Erfan Ahmadi contributed to annotation.*
Course repository: [maolama/LLM-Course-Spring2025](https://github.com/maolama/LLM-Course-Spring2025)

## Method

1. **Benchmark.** 367 English prompts across 10 task types: Facts, Reasoning, Math, STEM, Algorithmic/Coding, Extraction, Creative Writing, Counter-factual, Opinion-based, and Open-ended. The prompts were translated into Persian with Gemini (`codes/Translation.py`).
2. **Controlled answer variants.** For every prompt, an LLM (Claude 3.7) generates a set of Persian answers under strict length and content rules (`codes/ResponseGeneration.py`):

   | Variant | Description |
   |---|---|
   | `short_correct` | ≤ 100 words, correct |
   | `long_restricted` | 200–350 words, a paraphrase of the short answer with **no new information** |
   | `long_unrestricted` | 200–350 words, free to add examples and explanation |
   | `short_incorrect` / `long_incorrect` | contain subtle, realistic errors, with an explanation of each error |

3. **Pairwise judging.** Two judges, **DeepSeek-R1** and **Gemma 3 27B** (via OpenRouter), compare pairs of answers using an MT-Bench-style prompt that outputs `[[A]]`, `[[B]]` or `[[C]]` for a tie (`codes/Evaluator.py`). Every pair is judged **in both orders**. A verdict counts only if it is consistent across the swap; otherwise it is labeled as position bias or an unstable evaluation (`codes/utils/Evaluation_Aggregator.py`).

## Experiments

| # | Comparison | Question it answers |
|---|---|---|
| 0 | short vs. long (with extra information) | Baseline: do judges prefer length at all? |
| 1 | short vs. long (**same information**) | Is the preference for *length itself*? |
| 2 | Experiment 0 with an **English** judge prompt | Does the prompt language matter? |
| 3 | short correct vs. **long incorrect** | Does length override correctness? |

## Results
Judged on 108 questions (Algorithmic, Math, Creative Writing). The numbers count consistent verdicts only.

| Experiment | DeepSeek-R1: long / short | Gemma 3 27B: long / short |
|---|---|---|
| 0 — baseline | 91 / 4 | 104 / 0 |
| 1 — information parity | 57 / 24 | 69 / 18 |
| 2 — English prompt | 99 / 1 | 107 / 0 |
| 3 — long answer is wrong | 25 / 68 | 49 / 40 |

**Findings**
- Both judges strongly favor the longer answer, and they still do when the long answer only repeats the short one (Experiment 1).
- Switching the judging prompt to English does not reduce the bias.
- When the long answer contains factual errors, DeepSeek-R1 mostly picks the correct short answer. Gemma 3 still prefers the long, incorrect answer more often than not.
- Position bias is present but small: it is the most common reason a verdict is inconsistent.

## Repository Structure

```
Main.py                      # runs response generation and evaluation
codes/
  Translation.py             # EN → FA prompt translation (Gemini)
  ResponseGeneration.py      # controlled answer variants
  Evaluator.py               # pairwise judging experiments (both orders)
  GlobalVars.py              # generation and judge prompts
  Environment.py             # loads API settings from .env
  utils/                     # aggregation, consistency analysis, charts
data/
  questions/<category>/      # original + translated prompts
  responses/<model>/         # generated answer variants
  evaluation/<judge>/        # raw judge outputs and consistency labels
logs/                        # token and time usage
```

## Running
```bash
pip install -r requirements.txt
```
Create a `.env` file with `RESPONSE_GENERATION_MODEL`, `RESPONSE_GENERATION_API_URL` and `RESPONSE_GENERATION_API_KEY` (any OpenAI-compatible endpoint, e.g. OpenRouter). For translation, also add the `TRANSLATION_*` settings. Then:
```bash
python Main.py
```
