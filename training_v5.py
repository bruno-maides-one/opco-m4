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
    num_train_epochs=200,
    per_device_train_batch_size=16,
    #    use_cpu=True, # On reste sur CPU pour le test
)

# --- training via la class ---

trainer = modl.train(training_args)
# Resultat du training 80 epochs / 2 batch size : {'train_runtime': 147.4347, 'train_samples_per_second': 43.409, 'train_steps_per_second': 10.852, 'train_loss': 0.07630807340145111, 'epoch': 80.0}
# Resultat du training 200 epochs / 16 batch size : {'train_runtime': 219.9894, 'train_samples_per_second': 72.731, 'train_steps_per_second': 4.546, 'train_loss': 0.043269987106323245, 'epoch': 200.0}
modl.save()
