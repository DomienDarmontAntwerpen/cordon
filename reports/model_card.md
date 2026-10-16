# Quality gate model card

## What it is
The classifier behind each route's quality check: when routing sent a request to a cheaper
model, an answer scoring below the route's threshold is asked again of the strong model, once.
The thresholds belong to the routes - the benchmark proves each route *with* its check
(`evaluation/route_proof.py`). It catches the *signs* of a wrong answer; it cannot check a fact or a
calculation itself.

## Data
1197 real answers of the cheaper models (gpt-4.1-mini: 400, gpt-4o-mini: 797) to 997 questions with known
correct answers (AG News, GSM8K, MBPP, SST-2, golden set), of which 190 are wrong.
Questions the strong model also answered are left out: they prove the routes, and the check is
measured on them, so it never trains on them (the model file stores the ids it trained on).

Every answer carries the model's own confidence over the answer text
(OpenAI logprobs, measured before the benchmark's FINAL line, as a live answer has none).
Labels come from exact scoring against the known answer (summaries: an LLM judge's rating).
Written by the live benchmark and `evaluation.benchmark --collect` to `data/labelled_answers/`.

## Features
`quality.extract_features()`, 19 in all: the answer's shape (similarity to the question,
length, refusal, error, repetition, cut off), the kind of question, format and size (code block,
valid Python, share of digits), and the model's confidence. Without confidence (a provider
that returns no logprobs) the classifier cannot judge: such an answer goes to the strong model
(a route at threshold 0 uses the rules only and needs no confidence). Answers are featurized as an app
would receive them: the benchmark's own "FINAL:" line is removed first, so the gate never
learns a quirk of the test setup that real traffic does not have.

## Evaluation - 5-fold cross-validation grouped by question (never scored on a seen question)

| Candidate | AUC | Average precision (wrong answers) |
|---|---|---|
| LogisticRegression | 0.811 | 0.545 |
| RandomForest | 0.799 | 0.534 |
| GradientBoosting | 0.788 | 0.513 |

Selected: **LogisticRegression** (the simplest within 0.01 AUC of the best), refit on all answers.

What it has to beat:
- knowing only the kind of question: AUC 0.635
- the six answer-shape features only (the first gate): AUC 0.719

For comparison only (the thresholds that run are the routes'): a threshold per kind of question
that keeps 90% of its good answers (kinds with fewer than 50 good answers share 0.415)
catches, out-of-fold, **45% of wrong answers** while 90% of good answers are left alone.

Neural networks (an MLP on embeddings, a fine-tuned MiniLM transformer) and an LLM judge
were measured against this model in `notebooks/quality_gate_real_answers.ipynb`.

### Per kind of question (out-of-fold, each at its own threshold)

The check only ever acts where routing sends a cheaper model. One shared threshold would leave
some kinds nearly unguarded; that is why each route has its own, proven threshold.

| Kind | Answers | Wrong | AUC | Threshold | Wrong answers caught | Good left alone |
|---|---|---|---|---|---|---|
| chat | 175 | 56 | 0.864 | 0.127 | 55% | 90% |
| classification | 600 | 51 | 0.699 | 0.502 | 22% | 90% |
| code | 197 | 38 | 0.677 | 0.371 | 24% | 90% |
| extraction | 1 | 0 | - | 0.415 | - | 100% |
| reasoning | 224 | 45 | 0.873 | 0.414 | 76% | 90% |

### Noise and coverage
- 63 of the 699 questions gpt-4o-mini answered in more than one run got different labels:
  the same model answers differently from run to run, which limits how precise any figure here can be.
- The rule layer (empty, cut off, non-answer, refusal, error, invalid JSON) fired on none of these
  real answers - real models rarely produce such failures. It is tested on constructed cases
  (`tests/test_quality.py`), and is there for the day a provider does.

## Largest weights (standardized features; positive = more likely a good answer)

| Feature | Weight |
|---|---|
| confidence | +0.910 |
| digit_share | -0.849 |
| answer_words_log | -0.778 |
| repetition_ratio | -0.635 |
| qa_cosine | +0.313 |
| type_classification | +0.311 |
| type_chat | -0.252 |
| type_code | -0.239 |
| has_code | -0.239 |
| code_parses | -0.239 |

## Limits
- It reads signs, not truth: a wrong number in a confident, well-formed answer passes.
- Trained on the cheaper models' answers listed under Data, to the kinds of questions in the
  table above. On another model or kind of traffic it can be much weaker - that is measured, not
  assumed: `evaluation/route_proof.py` proves each route with the check on that route's own
  model. On gpt-4.1-mini's classification answers it stays barely better than chance (AUC about
  0.6, also after training on 400 of that model's own answers): a wrong label is usually a
  confident one. That route passes because gpt-4.1-mini itself is nearly as good as gpt-4o.
- It is never used on its own: a route is proven together with its check, and an answer
  without the model's confidence goes to the strong model.

## File integrity
SHA-256 of `src/models/model.pkl`: `c4ddf94cf8e271be6a47297f87870531630d7a365e32d1b3a8d3f8be6b970692` (recorded in `src/config/config.yaml`,
checked before every load).
