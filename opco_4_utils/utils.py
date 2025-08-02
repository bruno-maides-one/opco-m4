import torch
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from loguru import logger
from datasets import Dataset

class opco_4_utils:
    """
    Classe destiné a manipuler un model de type CLM (Type GPT)

    Features :

    * Training
    * Prediction
    """

    def __init__(self, model_name: str, device: str = None):
        """
        Constructeur qui initialise le model et le tokenizer a partir du nom de model fourni.

        :param model_name: Nom du model
        :type model_name: str
        :param device: Device à utiliser pour les predictions et le training. Si non spécifié la classe va essayer de
                       déterminer automatiquement le device les plus adapté
        :type device: str

        :rtype: opco_4_utils
        """
        # Init des properties
        self.initialized = False
        self.__column_name = "sample"
        self.__test_size = 0.2
        self.__max_length = 2048
        self.__dataset = None
        self.__tokenized_dataset = None
        self.__tokenized_dataset_train = None
        self.__tokenized_dataset_test = None
        self.__tokenized_dataset_train = None

        self.__model_name = model_name

        try:
            logger.info(f"Initialisation de la classe opco_4_utils")
            self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer {model_name} chargé")
            self.__model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info(f"Model {model_name} chargé")
        except Exception as e:
            logger.error(f"opco_4_utils Exception : {e}")
            raise e

        # Definition du device
        if device is None:
            self.detect_device()
            logger.info(f"Determination automatique du device : {self.__device}")
        else:
            self.__device = device
            logger.info(f"Device forcé : {self.__device}")

        # Paramétrage du tokenizer et du model
        self.__tokenizer.pad_token = self.__tokenizer.eos_token
        self.__model.config.pad_token_id = self.__tokenizer.pad_token_id

        # Déplacement des eléments sur le bon device
        self.__model.to(self.__device)
        logger.info(f"Model déployer sur le device : {self.__device}")

        # On a fini l'initialisation
        self.initialized = True

    #
    # Setters/Getters
    #

    # Device
    def set_device(self, device: str) -> None:
        """
        défini le device
        :param device:
        :rtype: None:
        """
        self.__device = device

    def get_device(self) -> str:
        """
        :return: La chaine codant le device utilisé
        :rtype: str
        """
        return self.__device

    device = property(get_device, set_device)

    # Model
    def get_model(self) -> AutoModelForCausalLM:
        """
        :return: L'instance du model utilisé par la classe
        :rtype: AutoModelForCausalLM
        """
        return self.__model

    model = property(get_model)

    # Tokenizer
    def get_tokenizer(self) -> AutoTokenizer:
        return self.__tokenizer

    tokenizer = property(get_tokenizer)

    # dataset
    def get_dataset(self) -> Dataset:
        return self.__dataset

    dataset = property(get_dataset)

    # tokenized_dataset

    def get_tokenized_dataset(self) -> Dataset:
        return self.__tokenized_dataset

    tokenized_dataset = property(get_tokenized_dataset)

    # tokenized_dataset_train

    def get_tokenized_dataset_train(self) -> Dataset:
        return self.__tokenized_dataset_train

    tokenized_dataset_train = property(get_tokenized_dataset_train)

    # tokenized_dataset_test

    def get_tokenized_dataset_test(self) -> Dataset:
        return self.__tokenized_dataset_test

    tokenized_dataset_test = property(get_tokenized_dataset_test)

    #
    # Methodes utiles
    #

    def detect_device(self) -> str:
        """
        Detect le device et le définie comme device utilisé dans l'instance.
        :return: La chaine décrivant le device detecté
        :rtype: str
        """
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.__device

    def tokenize_function(self, source: str | list[str], max_length: int = 2048):
        """
        Tokenise la chaine `source`
        :param source: Chaine à tokenizer
        :type source: str
        :param max_length: Longueur maximal de la séquence de tokens
        :type max_length: int

        :return: La séquence de tokens
        :rtype: list
        """
        return self.__tokenizer(source[self.__column_name], truncation=True, max_length=max_length)




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
