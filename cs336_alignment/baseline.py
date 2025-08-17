# uv run cs336_alignment/baseline.py

from vllm import LLM, SamplingParams
import json
from collections.abc import Callable
from typing import List
import pandas as pd
import torch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn 
# print(f"Torch Cude is available: {torch.cuda.is_available()}")

with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def save_to_jsonl(data, output_path):
    """
    Save a list of dictionaries to a JSONL file.

    Args:
        data (list[dict]): List of dictionaries to save.
        output_path (str): File path where the JSONL file will be saved.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# runs inference on the prompts 
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams, 
    answers: List[str]
    ) -> None:  
    """ Evaluate a language model on a list of prompts, compu"""
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    final_result = []
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        gen_text = output.outputs[0].text
        rewards = reward_fn(gen_text, answer)
        curr_res = {"prompt": prompt, "generated_text": gen_text, "correct_answer": answer, "rewards": rewards}
        final_result.append(curr_res)
    return final_result


# loads the dataset, serialize the results, and store it in the output path
def main(model_name="Qwen/Qwen2.5-Math-1.5B", dataset_path="./data/gsm8k/test.jsonl", output_path="./cs336_alignment/outputs/gsm8k/test.jsonl", temperature=1.0, top_p=1.0, max_tokens=1024):
    # load dataset 
    with open(dataset_path, "r") as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    print(f"Total number of dataset loaded for inference is {len(prompt_data)}")
    
    # preprocess the dataset 
    prompts = [R1_ZERO_PROMPT.format(question=pd['question']) for pd in prompt_data]
    gt_answer = [pd['answer'] for pd in prompt_data]
    
    # load model 
    llm = LLM(model=model_name)
    
    # initialize sampling params 
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    import ipdb; ipdb.set_trace()
    results = evaluate_vllm(llm, reward_fn=r1_zero_reward_fn, prompts=prompts[:3], eval_sampling_params=sampling_params, answers = gt_answer[:3])

    results = evaluate_vllm(llm, reward_fn=r1_zero_reward_fn, prompts=prompts, eval_sampling_params=sampling_params, answers = gt_answer)
    
    # store the result to output path
    save_to_jsonl(results, output_path)


if __name__ == "__main__":
    main()