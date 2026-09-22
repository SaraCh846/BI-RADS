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
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0, input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response
