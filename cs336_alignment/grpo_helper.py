from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
local_directory = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
model.save_pretrained(local_directory)
tokenizer.save_pretrained(local_directory)

def compute_group_normalized_rewards(
    reward_fn, 
    rollout_responses, 
    repeated_ground_truths, 
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    # calculate rewards 
    rewards_list = [reward_fn(pred, gold) for pred, gold in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.Tensor([rwd["reward"] for rwd in rewards_list])
    rewards = raw_rewards.reshape(-1, group_size)
    mean_rewards = rewards.mean(dim=-1, keepdim=True)
    group_normalized_reward = rewards - mean_rewards

    metadata = {"mean": mean_rewards}
    if normalize_by_std:
        rewards_std = torch.std(rewards, dim=-1, keepdim=True)
        metadata['std'] =  rewards_std
        group_normalized_reward = group_normalized_reward / (rewards_std + advantage_eps)

    advantage = group_normalized_reward.flatten() # rollout batch_size
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





