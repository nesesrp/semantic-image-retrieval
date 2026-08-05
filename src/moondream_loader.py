from transformers import AutoModelForCausalLM

_model = None


def load_moondream():
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
        )
       
        _model = _model.float()
    return _model
