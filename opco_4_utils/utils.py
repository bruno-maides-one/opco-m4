import torch
from torch import device
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from loguru import logger

class opco_4_utils:
    def __init__(self, model_name: str, device: [Optional[str]] = None):
        """
        Constructeur qui initialise le model et le tokenizer a partir du nom de model fourni
        :param model_name:
        :param device:
        """
        self.initialized = False

        try:
            logger.info(f"Initialisation de la classe opco_4_utils")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer {model_name} chargé")
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info(f"Model {model_name} chargé")
        except Exception as e:
            logger.error(f"opco_4_utils Exception : {e}")
            raise e

        # Definition du device
        if device is not None:
            self.detect_device()
            logger.info(f"Determination automatique du device : {self._device}")
        else:
            self._device = device
            logger.info(f"Device utilisé : {self._device}")

        # Paramétrage du tokenizer et du model
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.pad_token_id

        # Déplacment des eléments sur le bon device

        # On a fini l'initialisation
        self.initialized = True

    #
    # Setters/Getters
    #

    # Device
    def set_device(self, device: str) -> None:
        self._device = device

    def get_device(self) -> str:
        return self._device

    device = property(get_device, set_device)

    # Model
    def get_model(self) -> AutoModelForCausalLM:
        return self._model

    def set_model(self, model: AutoModelForCausalLM) -> None:
        self._model = model

    model = property(get_model)

    # Tokenizer
    def get_tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def set_tokenizer(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer

    tokenizer = property(get_tokenizer)

    #
    # Methodes utiles
    #

    def detect_device(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device


    def __old_init_model(model_name, device=None, pad_token=None):
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

    def do_predict(self, model,
                   prompt: str,
                   max_length: int = 200,
                   temperature: float = 0.7,
                   top_k: int = 40,
                   top_p: float = 0.9) -> str:

        logger.trace(f"Prompte : {prompt}")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        logger.trace(f"Input ids : {input_ids}")
        generated_ids = model.generate(input_ids,
                                       max_length=max_length,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p,
                                       do_sample=True)
        logger.trace(f"ids: {generated_ids[0]}")
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
