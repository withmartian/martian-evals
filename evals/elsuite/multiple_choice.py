from typing import Optional
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from pydantic import BaseModel
import re

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.formatting import make_abc
from evals.record import RecorderBase


class Sample(BaseModel):
    question: str
    answers: list[str]
    label: int
    kshot: Optional[str] = None
    subject: Optional[str] = None


def get_choices():
    return ["A", "B", "C", "D"]


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(example, include_answer=True):
    prompt = example["question"]
    for j in range(4):
        prompt += "\n{}) {}".format(get_choices()[j], example["choices"][j])
    prompt += "\nWhich one of the four choices completes the question correctly? Print only a single choice from \"A\" or \"B\" or \"C\" or \"D\" (without quotes or punctuation) corresponding to the correct answer without explanation. For example, if the answer is \"A\", then the output should be:\nAnswer:\nA\n\n Answer:\n"
    if include_answer:
        prompt += "{}\n\n".format(get_choices()[example["answer"]])
    return prompt


def format_example_hellaswag(example, include_answer=True):
    prompt = example["ctx"]
    for j in range(4):
        prompt += "\n{}. {}".format(get_choices()[j], example["endings"][j])
    prompt += "\nWhich one of the four choices completes the question correctly? Only output A, B, C, or D without explanation. Choice: "
    if include_answer:
        prompt += "{}\n\n".format(get_choices()[int(example["label"])])
    return prompt


def gen_prompt(train_dataset, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    for i in range(k):
        prompt += format_example(next(train_dataset))
    return prompt


def gen_prompt_hellaswag(train_dataset, k=-1):
    prompt = "Choose the most plausible continuation for the story. \n\nThe following are example stories and continuations (with answers).\n\n"
    for i in range(k):
        prompt += format_example_hellaswag(next(train_dataset))
    return prompt


def get_dataset(url: str) -> list[Sample]:
    parsed = urlparse(url)
    if parsed.scheme == "hf":
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        path = parsed.netloc
        if path == "hellaswag":
            if query.get("split") == "validation":
                query["split"] = "validation[:1%]"
        else:
            if query.get("split") == "validation":
                query["split"] = "test"
            subject = query.get("name")
            kshot_query = {}
            for key in query:
                if key != "split":
                    kshot_query[key] = query[key]
            kshot_query["split"] = "dev"
            kshot_data = load_dataset(path, **kshot_query)
            kshot_prompt = gen_prompt(iter(kshot_data), subject=subject, k=5)

        dataset = load_dataset(path, **query)

        if path == "hellaswag":
            return [
                Sample(
                    question=sample["ctx"],
                    answers=sample["endings"],
                    label=int(sample["label"]),
                    kshot="Choose the most plausible continuation for the story. \n\n"
                )
                for sample in dataset
            ]
        elif path == "hendrycks_test":
            return [
                Sample(
                    question=sample["question"],
                    answers=sample["choices"],
                    label=sample["answer"],
                    subject=subject,
                    kshot="Which one of the four choices completes the question correctly? Print only a single choice from \"A\" or \"B\" or \"C\" or \"D\" (without quotes or punctuation) corresponding to the correct answer without explanation. For example, if the answer is \"A\", then the output should be:\nAnswer:\nA\n\n",
                )
                for sample in dataset
            ]

    raise ValueError(f"Unknown question dataset {url}")


def fuzzy_match(s1, s2):
    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def exact_match(s1, correct_answer):
    if s1 == "" or correct_answer == "":
        return s1 == correct_answer
    # Strip out all variations of answer
    s1 = s1.replace("Answer", "")
    s1 = s1.replace("answer", "")
    correct_permutations = [correct_answer,
                            " " + correct_answer,
                            correct_answer + " ",
                            " " + correct_answer + " ",
                            correct_answer + ")",
                            correct_answer + "\n",
                            "\n" + correct_answer,
                            "\n" + correct_answer + "\n"]
    # I want regex for matching A,B,C,D and optionally a parentheses, space, or newline next to it
    # Regex for the correct_answer with only spaces, newlines, and parentheses allowed next to it
    # Regex for only matchines spaces, parentheses, and newlines
    regex = r"[\s\n\(\)]{,1}[^a-z]" + correct_answer + r"[\s\n\(\)]{,1}[^a-z]"
    output = re.search(regex, " " + s1 + " ")
    if output:
        return True
    else:
        return s1 in correct_permutations


class MultipleChoice(evals.Eval):
    def __init__(
            self,
            completion_fns: list[CompletionFn],
            dataset: str,
            *args,
            instructions: Optional[str] = "",
            **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MultipleChoice only supports one completion fn"
        self.dataset = dataset
        self.instructions = instructions
        if "hellaswag" in self.dataset:
            dataset = load_dataset("hellaswag", split="train")
            self.instructions = gen_prompt_hellaswag(iter(dataset), k=10)
        else:
            self.instructions = self.instructions

    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )

        prompt = (
                sample.kshot
                + "Please answer with the letter of the correct answer."
                + "\n\n"
                + "Question: "
                + sample.question
                + "\n\n"
                + options
                + "\n\nPrint only a single choice from \"A\" or \"B\" or \"C\" or \"D\" without explanation. Answer: "
        )
        prompt = (
                self.instructions
                + "\nPlease answer with the letter of the correct answer."
                + "\n\n"
                + sample.question
                + "\n"
                + options
                #+ "\n\nAnswer:"
        )
        # for j in range(len(sample.answers)):
        #    prompt += "\n{}) {}".format(get_choices()[j], sample.answers[j])
        #prompt += "\nAnswer:"
        prompt += "\nPrint only a single choice  from \"A\" or \"B\" or \"C\" or \"D\" without explanation. Answer:"
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=8,
            # top_p=1.0,
            # top_k=50,
        )
        sampled = result.get_completions()[0]

        match = exact_match(sampled, correct_answer)

        evals.record.record_match(
            match,
            expected=correct_answer,
        )
        evals.record.record_metrics(
            accuracy=float(match),
        )
        # evals.record_and_check_match(
        #    prompt=prompt,
        #    sampled=sampled,
        #    expected=correct_answer,
        # )

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
