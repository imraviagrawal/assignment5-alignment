from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
local_directory = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
model.save_pretrained(local_directory)
# tokenizer.save_pretrained(local_directory)


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")):
    # tokenize input string W/ EOD 
    # Tokenize Output String W/ EOD     
    prompt_tokens = tokenizer(prompt_strs)["input_ids"]
    output_tokens = tokenizer(output_strs)["input_ids"]

    batch_sz = len(prompt_tokens)
    p_o_combined = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
    max_p_o_len = max(p_o_combined)

    input_ids = torch.zeros((batch_sz, max_p_o_len-1))
    labels = torch.zeros((batch_sz, max_p_o_len-1))
    response_mask = torch.zeros((batch_sz, max_p_o_len-1), dtype=torch.bool)

    for i, (p, o) in enumerate(zip(prompt_tokens, output_tokens)):
        # iterate on each of the p o pair and construct the batch 
        p_o_concat = torch.tensor(p + o)
        concat_len = len(p_o_concat)
        p_o_concat_padded = F.pad(p_o_concat, (0, max_p_o_len - concat_len), "constant", tokenizer.eos_token_id)

        input_ids[i] = p_o_concat_padded[:-1]
        labels[i] = p_o_concat_padded[1:]

        o_start = len(p) - 1
        o_end = concat_len - 1
        response_mask[i, o_start:o_end] = True
    return {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.tensor:
    # compute softmax 
    # p_numerator = torch.exp(logits)
    # p_denom = torch.sum(p_numerator, dim=-1, keepdim=True)

    # log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # summand = (p_numerator/p_denom)*log_prob
    # return -torch.sum(summand, dim=-1) # b, s
    summand = F.softmax(logits, dim=-1)* (logits - torch.logsumexp(logits, dim=-1, keepdim=True))
    return -torch.sum(summand, dim=-1)

def get_response_log_probs(model, input_ids, labels, return_token_entropy) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
    log_probs = log_probs.squeeze(-1)
    ret_dict = {}
    ret_dict['log_probs'] = log_probs

    if return_token_entropy:
        ret_dict['token_entropy'] = compute_entropy(logits)
    
    return ret_dict

def masked_normalize(tensor, mask, normalize_constant, dim=None):
    # if dim == None: dim = -1
    # masked_tensor = torch.masked_select(tensor, mask)
    tensor_sum = torch.sum(tensor*mask, dim=dim)
    tensor_normalize = tensor_sum/normalize_constant
    return tensor_normalize

def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant):
    loss = (-masked_normalize(policy_log_probs, response_mask, normalize_constant, -1)).mean()
    loss /= gradient_accumulation_steps
    loss.backward()
    loss_metadata = {
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'normalize_constant': normalize_constant
    }

    return (loss, loss_metadata)