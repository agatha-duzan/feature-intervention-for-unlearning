import lm_eval
import os
import nest_asyncio
from lm_eval.models.openai_completions import OpenAIChatCompletion
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
from typing import Any, Dict, List, Optional, Tuple, Union

## Usage:
##     variant_eval(
##       tasks=["mmlu_flan_n_shot_generative_high_school_computer_science"],
##       variant=goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct"),
##       num_fewshot=0,
##       num_concurrent=10,
##       api_key=...,
##     )
## Extra parameters that can be passed to lm_eval.simple_evaluate like "limit=10" can also be passed.
##
## To use wmdp, the tasks are: "regexp_wmdp_cyber", "regexp_wmdp_chem", "regexp_wmdp_bio"
##
def variant_eval(tasks, variant, api_key, num_concurrent=1, max_retries=10, debug=False, max_gen_toks=256, **kwargs):
    os.environ['OPENAI_API_KEY'] = api_key
    lm_obj = VariantModel(base_url="https://api.goodfire.ai/api/inference/v1/chat/completions", model=variant.base_model, controller=variant.controller.json(), num_concurrent=num_concurrent,
                          max_retries=max_retries, debug=debug, max_gen_toks=max_gen_toks)

    task_manager = lm_eval.tasks.TaskManager(include_path="tasks")

    evaluation_tracker=EvaluationTracker(output_path="variant-output")

    nest_asyncio.apply()
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        model_args=[],
        task_manager=task_manager,
        apply_chat_template=True,
        evaluation_tracker=evaluation_tracker,
        tasks=tasks,
        **kwargs
    )
    samples = results.pop("samples")
    evaluation_tracker.save_results_aggregated(
      results=results, samples=samples
    )
    for task_name, config in results["configs"].items():
      evaluation_tracker.save_results_samples(
        task_name=task_name, samples=samples[task_name]
      )
    print(make_table(results))
    return results

class VariantModel(OpenAIChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        controller=None,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        self.controller = controller
        self.debug = debug

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        **kwargs,
    ) -> dict:
        if self.debug:
            print(messages)

        payload = super()._create_payload(
            messages=messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, **kwargs
        )

        # Add the `controller` field to the payload if it's not None
        if self.controller is not None:
            payload["controller"] = self.controller

        return payload
