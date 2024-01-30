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
    prompt = "Please answer with the letter of the correct answer.\n\n"
    prompt += example["question"]
    for j in range(4):
        prompt += "\n{}) {}".format(get_choices()[j], example["choices"][j])
    prompt += "\nPrint only a single choice  from \"A\" or \"B\" or \"C\" or \"D\" without explanation. Answer:\n"
    if include_answer:
        prompt += "{}\n\n".format(get_choices()[example["answer"]])
    return prompt


def format_example_hellaswag(example, include_answer=True):
    prompt = example["ctx"]
    for j in range(4):
        prompt += "\n{}) {}".format(get_choices()[j], example["endings"][j])
    prompt += "\nPrint only a single choice  from \"A\" or \"B\" or \"C\" or \"D\" without explanation.\nAnswer:\n"
    if include_answer:
        prompt += "{}\n\n".format(get_choices()[int(example["label"])])
    return prompt



def gen_prompt(train_dataset, subject, k=-1):
    prompt = "" #f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    for i in range(k):
        prompt += format_example(next(train_dataset))
    return prompt


def gen_prompt_hellaswag(train_dataset, k=-1):
    prompt = ""
    for i in range(k):
        prompt += format_example_hellaswag(next(train_dataset))
    return prompt


def format_example_winogrande(example, include_answer=True):
    prompt = "Please answer with the letter of the correct answer.\n\n"
    prompt += example["sentence"]
    prompt += "\n{}) {}".format("A", example["option1"])
    prompt += "\n{}) {}".format("B", example["option2"])
    prompt += "\nPrint only a single choice  from \"A\" or \"B\" without explanation. Answer:\n"
    if include_answer:
        prompt += "{}\n\n".format(get_choices()[int(example["answer"])-1])
    return prompt


def gen_prompt_winogrande(train_dataset, k=-1):
    prompt = "" #f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    for i in range(k):
        prompt += format_example_winogrande(next(train_dataset))
    return prompt


def get_dataset(url: str) -> list[Sample]:
    parsed = urlparse(url)
    if parsed.scheme == "hf":
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        path = parsed.netloc
        if path == "hellaswag":
            if query.get("split") == "validation":
                query["split"] = "validation"
            kshot_query = {}
            for key in query:
                if key != "split":
                    kshot_query[key] = query[key]
            kshot_query["split"] = "train"
            kshot_data = load_dataset(path, **kshot_query)
            kshot_prompt = gen_prompt_hellaswag(iter(kshot_data), k=5)
        elif path == "hendrycks_test":
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
        else:
            if query.get("split") == "validation":
                query["split"] = "validation"
            subject = query.get("name")
            kshot_query = {}
            for key in query:
                if key != "split":
                    kshot_query[key] = query[key]
            kshot_query["split"] = "train"
            kshot_data = load_dataset(path, **kshot_query)
            kshot_prompt = gen_prompt_winogrande(iter(kshot_data), k=5)

        dataset = load_dataset(path, **query)

        if path == "hellaswag":
            return [
                Sample(
                    question=sample["ctx"],
                    answers=sample["endings"],
                    label=int(sample["label"]),
                    kshot=kshot_prompt,
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
                    kshot=kshot_prompt # "Which one of the four choices completes the question correctly? Print only a single choice from \"A\" or \"B\" or \"C\" or \"D\" (without quotes or punctuation) corresponding to the correct answer without explanation. For example, if the answer is \"A\", then the output should be:\nAnswer:\nA\n\n",
                )
                for sample in dataset
            ]
        elif path == "winogrande":
            return [
                Sample(
                    question=sample["sentence"],
                    answers=[sample["option1"], sample["option2"]],
                    label=int(sample["answer"])-1, # A or B, instead of 1 or 2
                    kshot=kshot_prompt # "Which one of the four choices completes the question correctly? Print only a single choice from \"A\" or \"B\" or \"C\" or \"D\" (without quotes or punctuation) corresponding to the correct answer without explanation. For example, if the answer is \"A\", then the output should be:\nAnswer:\nA\n\n",
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
            self.instructions = "" #"Choose the most plausible continuation for the story. \n\n"

    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )
        prompt = (
                sample.kshot
                + "Please answer with the letter of the correct answer.\n\n"
                + sample.question
                + "\n"
                + options
                + "\nPrint only a single choice  from \"A\" or \"B\" or \"C\" or \"D\" without explanation. Answer:"
        )
        if "hellaswag" in self.dataset:
            prompt = (
                sample.kshot
                + sample.question
                + "\n"
                + options
                + "\nPrint only a single choice  from \"A\" or \"B\" or \"C\" or \"D\" without explanation.\nAnswer:"
            )
        elif "winogrande" in self.dataset:
            prompt = (
                    sample.kshot
                    + sample.question
                    + "\n"
                    + options
                    + "\nPrint only a single choice  from \"A\" or \"B\" without explanation. Answer:"
            )

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=2,
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
