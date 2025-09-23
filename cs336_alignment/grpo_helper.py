from transformers import AutoTokenizer, AutoModel
import torch
from typing import Literal
import torch.nn.functional as F
from math_verify import parse, verify
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
# model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
# local_directory = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
# model.save_pretrained(local_directory)
# tokenizer.save_pretrained(local_directory)

def compute_group_normalized_rewards(
    reward_fn, 
    rollout_responses, 
    repeated_ground_truths, 
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    # calculate rewards 
    parsed_answer = []
    for res in rollout_responses:
        # try: 
        #     parsed_answer.append(parse(res)[1])
        # except:
        parsed_answer.append(res)
    rewards_list = [reward_fn(pred, gold) for pred, gold in zip(parsed_answer, repeated_ground_truths)]
    raw_rewards = torch.Tensor([rwd["reward"] for rwd in rewards_list])
    rewards = raw_rewards.reshape(-1, group_size)
    mean_rewards = rewards.mean(dim=-1, keepdim=True)
    group_normalized_reward = rewards - mean_rewards
    metadata = {"mean_rewards": mean_rewards}
    if normalize_by_std:
        rewards_std = torch.std(rewards, dim=-1, keepdim=True)
        metadata['std'] =  rewards_std
        group_normalized_reward = group_normalized_reward / (rewards_std + advantage_eps)

    advantage = group_normalized_reward.flatten() # rollout batch_size
    mean_advantage = advantage.mean(dim=-1, keepdim=True)
    metadata["mean_advantage"] = mean_advantage
    return (advantage, raw_rewards, metadata)

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor,policy_log_probs: torch.Tensor,) -> torch.Tensor:
    seq_len = policy_log_probs.shape[1]
    adv = raw_rewards_or_advantages.repeat_interleave(seq_len, dim=1)
    return -1 * adv * policy_log_probs

def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    seq_len = policy_log_probs.shape[1]
    advantages = advantages.repeat_interleave(seq_len, dim=1)    
    ratio = torch.exp(policy_log_probs - old_log_probs)
    lhs = ratio*advantages
    rhs = torch.clamp(ratio, min = 1 - cliprange, max = 1 + cliprange)*advantages
    loss = -1 * torch.min(lhs, rhs)
    metadata = {"lhs": lhs, "rhs": rhs, "clipped": rhs < lhs}
    return (loss, metadata)

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if loss_type == "no_baseline":
        assert raw_rewards != None, "Raw Rewards are required for the baseline"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    
    if loss_type == "reinforce_with_baseline":    
        assert advantages != None, "Advantages are required for the baseline or clip loss"
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    if loss_type == "grpo_clip":
        assert old_log_probs != None, "Old Log Probs needed for grpo loss"
        assert advantages != None, "Advantages are required for the baseline or clip loss"
        assert cliprange != None, "Clip range is need for grpo clip loss"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
    ) -> torch.Tensor:
    mask = mask.to(tensor.dtype) 
    masked = tensor * mask # project to same dtpye as input
    masked_sum = masked.sum(dim=dim)
    masked_count = mask.sum(dim=dim)
    return masked_sum / masked_count

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # get per token loss 
    loss_t, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)


    # masked loss scaler / example
    loss_e = masked_mean(loss_t, response_mask, dim=1) # per example mean (batch, 1)

    # single batch loss, gradient accumulation 
    loss = loss_e.mean()/gradient_accumulation_steps

    # backpropogate
    loss.backward()

    return loss, metadata


