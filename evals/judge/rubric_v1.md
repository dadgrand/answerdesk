# Answer rubric — `rubric-v1`

Prompt version: `judge-rubric-1`. Rubric version: `rubric-v1`.
The executable copy of this text lives in `src/answerdesk/eval/judge.py` (`RUBRIC_TEXT`) and a
test asserts that the two do not drift.

You grade one customer-support answer. You are not a copy editor and you are not a
salesperson: style, length, tone and confidence are irrelevant to this rubric.

For each claim in the answer:

1. **Supported (5 or 4).** The cited passage contains the claim's fact, and the claim copies
   identifiers, numbers, dates, fees and product names exactly as the passage writes them.
2. **Partially supported (3).** The cited passage implies the claim but a specific number,
   date or identifier is missing or rounded.
3. **Unsupported (1 or 2).** The passage does not state the claim, contradicts it, or the
   claim adds a fact the passage never mentions.

Anchor every judgement in the passage text, not in what you believe about banks, cards or
telecoms. If the answer correctly declares that the evidence is insufficient, that is a pass —
refusing is a correct answer, an invented fact is a defect.

## How the harness uses this

* claims are graded **one at a time** against the passage they cite (`context`), never against
  the answer as a whole — whole-answer grading against a one-sentence reference scores length;
* the per-answer rubric score is the mean of its claim scores; a pass is `≥ 4`;
* a reference answer is passed to the judge only when the corpus can support it
  (`metadata.reference_verifiable`); otherwise the rubric runs context-only and the diagnostic
  `rubric_reference_overlap` cell reports the gap;
* pairwise comparisons run the same rubric in both orders and average (`judge.compare_pair`);
* the judged metric is reportable for a language only when judge–human κ ≥ 0.60 for it.

## Why these anchors

Verbosity and self-preference are the two documented LLM-judge biases most likely to move a
support-quality number; both are addressed structurally instead of with wording ("never reward
length or confidence" is in the prompt, but the rubric's unit is the claim, which has no
length to reward).
