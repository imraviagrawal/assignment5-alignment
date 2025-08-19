
from transformers import AutoTokenizer

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")):
    # tokenize input string W/ EOD 
    # Tokenize Output String W/ EOD 

    # create mask where we split the tokenized input on EOD and add both together
    tokenizer_prompts = tokenizer(prompt_strs)
    p_input_ids, p_attn_mask = tokenizer_prompts['input_ids'], tokenizer_prompts['attention_mask'], #List[List], List[List]
    tokenized_outputs = tokenizer(output_strs)
    o_input_ids, o_attn_mask = tokenized_outputs['input_ids'], tokenized_outputs['attention_mask']
    tokenized_prompt_ids = []
    attention_mask = []
    for i in range(len(tokenizer_prompts)):
        p_ids, p_am = p_input_ids[i], p_attn_mask[i]
        o_ids, o_am = o_input_ids[i], o_attn_mask[i]
        ids = p_ids + o_ids
        am = p_am*0 + o_am
        tokenized_prompt_ids.append(ids)
        attention_mask.append(am)
    return tokenized_prompt_ids, attention_mask