import numpy as np
from datasets import load_dataset

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.record import RecorderBase

def get_choices():
    return ["A", "B", "C", "D"]


def format_example(example, include_answer=True):
    prompt = example["question"]
    for j in range(4):
        prompt += "\n{}. {}".format(get_choices()[j], example["choices"]["text"][j])
    prompt += "\n"
    if include_answer:
        prompt += " {}\n\n".format(example["answerKey"])
    return prompt


def gen_prompt(train_dataset, k=-1):
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for i in range(k):
        prompt += format_example(next(train_dataset))
    return prompt

class FuzzyMatch(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "FuzzyMatch only supports one completion fn"
        self.max_tokens = max_tokens
        # Do it manually for each of the specific ones
        if "arc_challenge" in samples_jsonl:
            self.k_shot = load_dataset("ai2_arc", "ARC-Easy")
            self.kshot_prompt = "\n\n" + gen_prompt(iter(self.k_shot["train"]), k=25)
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        # Add the 25 examples to the prompt, if arc challenge in samples_jsonl
        if "arc_challenge" in self.samples_jsonl:
            # Only the prompt[0]["content"] needs to be updated
            prompt[0]["content"] += self.kshot_prompt
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        matches = [utils.fuzzy_match(sampled, correct_answer) for correct_answer in correct_answers]

        evals.record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(sampled, correct_answers),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
