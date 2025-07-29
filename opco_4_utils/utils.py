from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = None
initialized = False

def init_model(model_name, device=None):
    """
    Initialise le tokenizer et le model identifié par 'model_name'
    :param model_name:
    :return:
    """
    # Création du tokenniser
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Chargement du model
    model = AutoModelForCausalLM.from_pretrained(model_name)

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
               pad_token_id: int = 0):
#               pad_token_id: int = tokenizer.eos_token_id):

    if not initialized:
        raise RuntimeError('You need to call init_model first')

    print(f"Prompte : {prompt}\n----")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Input ids : {input_ids}")
    print("---")
    generated_ids = model.generate(input_ids,
                                   max_length=max_length,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   pad_token_id=pad_token_id,
                                   do_sample=True)
    print("----")
    print(generated_ids[0])
    print("----")
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
