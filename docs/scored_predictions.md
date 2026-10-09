# Exact counts and paired case analysis

`tools/analyze_scored_predictions.py` aggregates **already scored** per-question
predictions. It does not run a model, extract mathematical answers, judge
correctness, or replace the benchmark evaluator. This utility was added during
release preparation; it is not presented as the historical inference script.

Export the official scorer's decisions as JSONL, keeping the dataset question ID:

```json
{"id": "question-id", "correct": true}
```

Run each condition using the same dataset revision, prompts, image preprocessing,
generation configuration and scorer. Retain raw outputs and scorer configuration
privately for auditing; do not convert inference failures into guessed answers or
drop failed questions. The utility requires exactly 1,000 MathVista `testmini`
questions, 1,319 GSM8K `test` questions, or 2,700 full Video-MME questions. This
checks counts and cross-condition ID equality, not the authenticity of the IDs or
the correctness of upstream judgments.

```bash
python tools/analyze_scored_predictions.py --benchmark mathvista_testmini \
  --run original=results/original.jsonl \
  --run video_lora=results/video_lora.jsonl \
  --run math_rehearsal=results/math_rehearsal.jsonl
```

For GSM8K, select `--benchmark gsm8k_test`. Output includes exact integer counts,
fractions, percentage values rounded to six decimal places, and paired question
IDs in four groups: both correct, lost, recovered and both incorrect. This makes
it possible to select auditable cases across conditions without fabricating
model responses. Lost/recovered counts can reveal changes masked by nearly equal
aggregate accuracy; they do not by themselves establish a causal explanation or
statistical significance. Greedy decoding does not turn repeated evaluation of
one checkpoint into independent training-seed experiments.

Use [MathVista's official evaluator](https://github.com/lupantech/MathVista) and
the [GSM8K dataset](https://github.com/openai/grade-school-math) with the declared
evaluation protocol. No private question-level predictions are distributed here.
