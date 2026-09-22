from models import load_model, prompt_model


def test_model_prompt():
    model_id = "microsoft/Phi-3.5-mini-instruct"

    model, tokenizer = load_model(model_id)

    response = prompt_model(
        model,
        tokenizer,
        "Extract the breast composition from this report: "
        "The breasts are heterogeneously dense."
    )

    print(response)

    assert isinstance(response, str)
    assert len(response) > 0
  
