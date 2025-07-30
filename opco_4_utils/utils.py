from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
tokenizer = None
initialized = False

def init_model(model_name, device=None, pad_token=None):
    """
    Initialise le tokenizer et le model identifié par 'model_name'
    :param model_name:
    :return:
    """
    global tokenizer, initialized
    # Création du tokenniser
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Chargement du model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Definition du pad_token si pas fourni
    if pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        print("Tokenizer par defaut")
    else:
        tokenizer.pad_token = pad_token
        print("Tokenizer custom")


    if device is not None:
        model.to(device)

    initialized = True

    return tokenizer, model

def do_predict(model,
               prompt: str,
               max_length: int = 200,
               temperature: float = 0.7,
               top_k: int = 40,
               top_p: float = 0.9,
               pad_token: Optional[int] = None):
#               pad_token_id: int = tokenizer.eos_token_id):

    if not initialized:
        raise RuntimeError('You need to call init_model first')

    if pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = pad_token

    print(f"Prompte : {prompt}\n----")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Input ids : {input_ids}")
    print("---")
    generated_ids = model.generate(input_ids,
                                   max_length=max_length,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   pad_token_id=pad_token,
                                   do_sample=True)
    print("----")
    print(generated_ids[0])
    print("----")
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
