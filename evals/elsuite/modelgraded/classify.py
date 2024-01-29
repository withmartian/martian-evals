"""
Generic eval that uses a prompt + classification.
"""
import os
from collections import Counter
from random import Random
from typing import Any, Optional, Union
import copy

import evals
import evals.record
from evals.elsuite.modelgraded.classify_utils import classify, sample_and_concat_n_completions
from evals.elsuite.utils import PromptFn, scrub_formatting_from_prompt



class ModelBasedClassify(evals.Eval):
    def __init__(
        self,
        modelgraded_spec: str,
        *args,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        sample_kwargs: Optional[dict[str, Any]] = None,
        eval_kwargs: Optional[dict[str, Any]] = None,
        multicomp_n: Union[int, str] = 1,
        eval_type: Optional[str] = None,
        match_fn: Optional[str] = None,
        metaeval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # treat last completion_fn as eval_completion_fn
        self.eval_completion_fn = copy.deepcopy(self.completion_fns[-1])
        self.eval_completion_fn.model = "gpt-4-1106-preview"
        self.eval_completion_fn.api_base = "https://api.openai.com/v1"
        self.eval_completion_fn.api_key = os.environ.get("REAL_OPENAI_API_KEY",None)
        # Get k-shot values for GSM-8k, get the first 5 examples, and include them as
        if len(self.completion_fns) > 1:
            self.completion_fns = self.completion_fns[:-1]
        n_models = len(self.completion_fns)
        self.sample_kwargs = {"max_tokens": 1024}
        self.sample_kwargs.update(sample_kwargs or {})
        self.eval_kwargs = {"max_tokens": 1024}
        self.eval_kwargs.update(eval_kwargs or {})
        self.metaeval = metaeval
        self.modelgraded_spec_args = modelgraded_spec_args or {}
        self.eval_type = eval_type
        self.match_fn = match_fn
        if multicomp_n == "from_models":
            assert n_models > 1
            self.multicomp_n = n_models
        else:
            assert isinstance(multicomp_n, int)
            self.multicomp_n = multicomp_n
        if len(self.completion_fns) > 1:
            assert self.multicomp_n == n_models

        self.is_gsm8k = True if "grade-school-math" in self.samples_jsonl else False
        self.mg = self.registry.get_modelgraded_spec(modelgraded_spec)

    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        # process test_sample
        for k in self.mg.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])

        # run policy completions
        completions = {}
        for k, v in self.mg.input_outputs.items():
            if v in test_sample:  # test_sample already has completion, skip.
                continue
            if self.multicomp_n > 1:
                completion = sample_and_concat_n_completions(
                    self.completion_fns,
                    prompt=test_sample[k],
                    template_i=self.mg.output_template,
                    sample_kwargs=self.sample_kwargs,
                    n=self.multicomp_n,
                )
            else:
                # Add few shot to beginning of prompt for GSM-8k
                if self.is_gsm8k:
                    test_sample[k][0]["content"] += "\nThe following are examples of grade school math problems and answers:\n"
                    test_sample[k][0]["content"] += "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? \n6\n\nIf there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? \n5\n\nLeah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? \n39\n\nJason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n8\n\nShawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? \n9\n\n"
                get_input_completion = PromptFn(
                    test_sample[k], completion_fn=self.completion_fn, **self.sample_kwargs
                )
                completion, _ = get_input_completion()
            completions[v] = completion

        # run modelgraded eval
        metrics = {}
        choice, info = classify(
            mg=self.mg,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_kwargs,
            eval_type=self.eval_type,
            n=self.multicomp_n,
            match_fn=self.match_fn,
            format_kwargs={**completions, **test_sample, **self.modelgraded_spec_args},
        )
        metrics.update(dict(choice=choice, score=info["score"]))

        # run metaeval if requested
        if self.metaeval:
            assert "choice" in test_sample
            metrics["metascore"] = choice == test_sample["choice"]

        evals.record.record_metrics(**metrics)

        return choice

    def run(self, recorder):
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        record_metrics = {}

        all_sample_metrics = recorder.get_metrics()
        if not all_sample_metrics:
            return record_metrics

        # record the counts
        choices = [m["choice"] for m in all_sample_metrics]
        counts = dict(Counter(choices))
        record_metrics.update({f"counts/{k}": v for k, v in counts.items()})

        # record the scores
        scores = [m["score"] for m in all_sample_metrics if m["score"] is not None]
        if scores:
            record_metrics["score"] = sum(scores) / len(scores)
        metascores = [m["metascore"] for m in all_sample_metrics if "metascore" in m]
        if metascores:
            record_metrics["metascore"] = sum(metascores) / len(metascores)

        return record_metrics
