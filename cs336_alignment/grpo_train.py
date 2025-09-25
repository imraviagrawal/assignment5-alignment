from transformers import AutoTokenizer, AutoModel
import torch
from pprint import pprint
from tqdm import tqdm
from typing import Literal, List, Dict
import torch.nn.functional as F
from cs336_alignment.vllm_helper import *
from cs336_alignment.grpo_helper import *
from cs336_alignment.sft_helper import *
from cs336_alignment.baseline import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from vllm import SamplingParams

with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def load_dataset(json_path):
    # expects a JSON array: [{"question":..., "answer":...}, ...]
    data = json.load(open(json_path))
    prompts = [d["question"] for d in data]
    responses = [d["answer"] for d in data]
    return prompts, responses

def load_math_data(filepath: str) -> List[Dict[str, str]]:
    # expects jsonl: one JSON object per line
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


class TextDataset(Dataset):
    def __init__(self, json_path):
        # self.text_list = text_list
        # self.tokenizer = tokenizer
        # self.max_len = max_len
        prompts, responses = load_dataset(json_path)
        self.prompts = prompts
        self.responses = responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        return prompt, response


def generate_grpo_rollouts(vllm_model, 
                           question_batch, 
                           prompt_template, 
                           sampling_params: SamplingParams
                           ):
    questions = [d['question'] for d in question_batch]
    answers = [d['answer'] for d in question_batch]
    prompts = [prompt_template.format(question=q) for q in questions]

    # group response per prompt 
    outputs = vllm_model.generate(prompts, sampling_params)

    # Flatten results and create repeated structures
    rollout_responses = []
    repeated_prompts = []
    repeated_ground_truths = []

    for output, que, gt in zip(outputs, questions, answers):
         for completion in output.outputs:
             rollout_responses.append(completion.text)
             repeated_prompts.append(output.prompt)
             repeated_ground_truths.append(gt.split("####")[1].strip())
    return repeated_prompts, repeated_ground_truths, rollout_responses


def train_grpo(model_name, 
               n_grpo_steps: int = 200,
               learning_rate: float = 1e-5,
               advantage_eps: float = 1e-6,
               rollout_batch_size: int = 256,
               group_size: int = 8,
               sampling_temperature: float = 1.0,
               sampling_min_tokens: int = 4, # As in Expiter, disallow empty string responses,
               sampling_max_tokens: int = 1024,
               epochs_per_rollout_batch: int = 1, # On-policy
               train_batch_size: int = 256, # On-policy,
               gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100,
               gpu_memory_utilization: float = 0.85,
               loss_type: Literal[
                   "no_baseline",
                   "reinforce_with_baseline",
                   "grpo_clip","grpo_noclip"] = "reinforce_with_baseline",
                use_std_normalization: bool = True,
                cliprange = 0.2,
                eval_steps=256,
                length_normalize=False,
                n_eval=1024,
                eval_log_frequecy = 8
            ):
    print(f"Loss Type: {loss_type}")
    
    assert train_batch_size % gradient_accumulation_steps == 0, ("train_batch_size must be divisible by gradient_accumulation_steps")
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, ("rollout_batch_size must be divisible by group_size")
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, ("train_batch_size must be greater than or equal to group_size")
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    # load policy 
    print("Loading Policy")
    policy_device = "cuda:0"
    policy, tokenizer = init_policy(policy_device)

    # initilize vllm model 
    print("loading vllm model")
    llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", "cuda:1", 0)  # llm for inference 
    optim = torch.optim.AdamW(policy.parameters(), 
                              lr=learning_rate, 
                              weight_decay=0.0, 
                              betas=(0.9, 0.95)
                              )
    
    # we need to convert this to torch dataset and shuffle
    # training dataset: 
    print("loading training Dataset")
    train_dataset = load_math_data("data/gsm8k/train.jsonl")
    # train_dataset = TextDataset("data/gsm8k/train.jsonl")
    # train_dataloader = DataLoader(train_dataset, batch_size=micro_train_batch_size, shuffle=True)
    
    # validation 
    eval_dataset = load_math_data("data/gsm8k/test.jsonl")
    # validation_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True) # shuffle not needed 

    # first loop 
    print("starting training")
    train_step = 0
    eval_results = []
    for step in tqdm(range(n_grpo_steps)):
        # sample n prompt per roll out
        idx = np.random.choice(len(train_dataset), n_prompts_per_rollout_batch, replace=False)
        question_batch = [train_dataset[i] for i in idx]

        # set old policy as policy: 
        print("update old policy with policy w/ last updates")
        load_policy_into_vllm_instance(policy, llm) # change old policy with last policy

        if train_step % eval_log_frequecy == 0:
            # generate roll outs 
            eval_params = SamplingParams(temperature=1.0,  
                                        max_tokens=sampling_max_tokens, 
                                        min_tokens=sampling_min_tokens, 
                                        stop=["</answer>"], 
                                        include_stop_str_in_output=True, 
                                        )
            idx = np.random.choice(len(eval_dataset), n_eval, replace=False)
            eval_dataset_sampled = [eval_dataset[id] for id in idx]
            eval_prompts, eval_answers = [], []
            for d in eval_dataset_sampled: 
                eval_prompts.append(R1_ZERO_PROMPT.format(question=d["question"]))
                eval_answers.append(d["answer"].split("####")[1].strip())
            eval_out = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_params, eval_answers)
            gen_texts = [len(d["generated_text"]) for d in eval_out]
            # import ipdb; ipdb.set_trace()
            sum_rewards = sum([d['rewards']["reward"] for d in eval_out])
            sum_format_reward = sum([d['rewards']["format_reward"] for d in eval_out])
            sum_answer_reward = sum([d['rewards']["answer_reward"] for d in eval_out])
            answer_len = len(eval_out)
            avg_answer_len = np.mean(gen_texts)
            std_answer_len = np.std(gen_texts)
            eval_results.append(f"\n TS: {train_step}, AR {sum_rewards/answer_len}, AFR: {sum_format_reward/answer_len}, \n Avg Len: {avg_answer_len}, Std Len: {std_answer_len}")
            pprint(eval_results)

        # generate roll outs 
        sampling_params = SamplingParams(temperature=sampling_temperature, 
                                    max_tokens=sampling_max_tokens, 
                                    min_tokens=sampling_min_tokens, 
                                    n=group_size, 
                                    stop=["</answer>"], 
                                    include_stop_str_in_output=True, 
                                    )
        
        
        repeated_prompts, repeated_ground_truths, rollout_responses = generate_grpo_rollouts(llm, 
                                                                                             question_batch, 
                                                                                             R1_ZERO_PROMPT, 
                                                                                             sampling_params)
        # print_idx = 2
        # print(f"A: {repeated_ground_truths[print_idx]}, \n RO: {rollout_responses[print_idx]}")

        # r1_zero_reward_fn(rollout_responses[print_idx], repeated_ground_truths[print_idx])
        # import ipdb; ipdb.set_trace()
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn, 
            rollout_responses, 
            repeated_ground_truths, 
            group_size,
            advantage_eps,
            normalize_by_std = use_std_normalization,
        )
        # import ipdb; ipdb.set_trace()
        if train_step % eval_log_frequecy == 0:
            print(f'Raw Rewards {raw_rewards.reshape(-1, group_size)}, \n')
            print(f"Advantages: {advantages.reshape(-1, group_size)}")
        advantages = advantages.to(policy_device)
        raw_rewards = raw_rewards.to(policy_device)
        print(f"Train Step: {train_step} Mean Rewards {raw_rewards.mean()}, mean advantage {advantages.mean()}")

        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses)
        input_ids_tensor = torch.Tensor(tokenized['input_ids']).to(policy_device).long() # b, seq
        labels_tensor = torch.Tensor(tokenized['labels']).to(policy_device).long()
        mask_tensor = torch.tensor(tokenized['response_mask']).to(policy_device)

        # get old log probs OOM 
        policy.eval()
        with torch.no_grad():
            old_policy_log_probs = []
            for idx in range(0, len(input_ids_tensor), micro_train_batch_size):
                old_policy_log_probs.append(get_response_log_probs(policy, input_ids_tensor[idx : idx + micro_train_batch_size], labels_tensor[idx : idx + micro_train_batch_size], return_token_entropy=True)['log_probs'])
            old_policy_log_probs = torch.cat(old_policy_log_probs, dim=0)

        for epoch in range(epochs_per_rollout_batch):
            optim.zero_grad()
            rollout_batch_loss = 0

            for idx in range(0, len(repeated_prompts), micro_train_batch_size):
                input = input_ids_tensor[idx : idx + micro_train_batch_size]
                label = labels_tensor[idx : idx + micro_train_batch_size]
                mask = mask_tensor[idx : idx + micro_train_batch_size]
                advantage = advantages[idx : idx + micro_train_batch_size]
                raw_reward = raw_rewards[idx : idx + micro_train_batch_size]
                old_log_prob = old_policy_log_probs[idx : idx + micro_train_batch_size]

                policy.train()
                policy_log_probs = get_response_log_probs(policy, input, label, True)
                token_entropy = policy_log_probs['token_entropy']
                policy_log_probs = policy_log_probs['log_probs']
                
                raw_reward = raw_reward.unsqueeze(1)
                advantage = advantage.unsqueeze(1)

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs, mask, gradient_accumulation_steps, loss_type, 
                    raw_reward, advantage, old_log_prob, cliprange, length_normalize, sampling_max_tokens
                )
                # import ipdb; ipdb.set_trace()
                rollout_batch_loss += loss.item()
            optim.step()
            train_step += 1
            print(f"Train Step: {train_step}, Roll out batch Loss: {rollout_batch_loss}")
            

if __name__ == "__main__":
    train_grpo("model", 
    loss_type="grpo_clip", 
    length_normalize=True, 
    use_std_normalization=False, 
    epochs_per_rollout_batch=4, 
    train_batch_size=256)



        






