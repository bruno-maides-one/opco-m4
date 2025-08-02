import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
from opco_4_utils import opco_4_utils
from data.dataset_training import dataset_training

# --- INIT ---
MODEL_NAME = "Salesforce/codegen-350M-mono"
print(f"--- Lancement du test final avec le modèle : {MODEL_NAME} ---")

# recupération du dataset dans data/dataset_training
# On passe par un dictionnaire au format "Prompt\ncode"
source_dataset = Dataset.from_dict(dataset_training)

print(source_dataset)

# --- Instanciation de la classe ---

modl = opco_4_utils(MODEL_NAME)

# Maintenant on recupère les dataset en utilisant la classe

modl.load_dataset(source_dataset)

# --- 5. Configuration de l'entraînement minimaliste ---
training_args = TrainingArguments(
    output_dir="./results_test_final",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    #    use_cpu=True, # On reste sur CPU pour le test
)

# --- training via la class ---

trainer = modl.train(training_args)

modl.save()
