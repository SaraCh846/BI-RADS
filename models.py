import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id):
    tokenizer_arguments = {}

    if model_id == "facebook/MobileLLM-1B":
        tokenizer_arguments["use_fast"] = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        **tokenizer_arguments
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer

def prompt_model(model, tokenizer, prompt, max_new_tokens=200):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    if tokenizer.chat_template is not None:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs.shape[-1]:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()
