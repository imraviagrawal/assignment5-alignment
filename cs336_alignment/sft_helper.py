from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
local_directory = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
model.save_pretrained(local_directory)
tokenizer.save_pretrained(local_directory)


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")):
    """
    prompt_strs: list[str]
    output_strs: list[str]
    """
    # tokenize input string W/ EOD 
    # Tokenize Output String W/ EOD   
    # import ipdb; ipdb.set_trace()  
    # print(type(tokenizer(prompt_strs)))
    prompt_ids = tokenizer(prompt_strs)['input_ids']
    output_ids = tokenizer(output_strs)['input_ids']

    p_o_len = [len(p) + len(o) for p, o in zip(prompt_ids, output_ids)]
    max_po_len = max(p_o_len)
    bsz = len(p_o_len)

    input_ids = torch.zeros((bsz, max_po_len-1))
    labels = torch.zeros((bsz, max_po_len-1))
    response_mask = torch.zeros((bsz, max_po_len-1), dtype=torch.bool)

    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        p_o_combined = torch.Tensor(p + o)
        len_p0 = len(p_o_combined)
        p_o_combined_pad = F.pad(p_o_combined, (0, max_po_len - len_p0), "constant", tokenizer.eos_token_id) # id 

        input_ids[i] = p_o_combined_pad[:-1]
        labels[i] = p_o_combined_pad[1:]
        
        start_o = len(p) - 1
        end_o = len_p0 - 1
        response_mask[i, start_o:end_o] = True


    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    summand = F.softmax(logits, dim=-1) * (logits - torch.logsumexp(logits, dim=-1, keepdim=True))
    return -1 * torch.sum(summand, dim=-1)

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits # get logits 
    log_p = F.log_softmax(logits, dim=-1)
    token_p = log_p.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = {"log_probs": token_p}
    
    if return_token_entropy:
        output['token_entropy'] = compute_entropy(logits)

    return output

def masked_normalize(tensor, mask, normalize_constant, dim=None):
    masked = tensor * mask.to(tensor.dtype)
    summed = torch.sum(masked, dim=dim)
    normalized = summed/normalize_constant
    return normalized

def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0):
    loss = -1 * masked_normalize(policy_log_probs, response_mask, normalize_constant, -1).mean()
    scaled_loss = loss/gradient_accumulation_steps
    scaled_loss.backward()
    with torch.no_grad():
        num_tokens = response_mask.sum().item()
        stats = {
            "loss": loss.detach().item(),
            "num_tokens": num_tokens,
            "normalize_constant": float(normalize_constant),
            "gradient_accumulation_steps": gradient_accumulation_steps
        }
    return (scaled_loss, stats)
