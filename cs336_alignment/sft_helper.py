from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
# local_directory = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
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
    p_numerator = torch.exp(logits)
    p_denom = torch.sum(p_numerator, dim=-1, keepdim=True)

    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    summand = (p_numerator/p_denom)*log_prob
    return -torch.sum(summand, dim=-1) # b, s