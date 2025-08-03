#
# Version de base pour arriver au training
#
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset

# --- INIT ---
MODEL_NAME = "Salesforce/codegen-350M-mono"
print(f"--- Lancement du test final avec le modèle : {MODEL_NAME} ---")

# --- 1. création du dataset ---
data = {
    "sample": [
        "Ecrit une fonction qui affiche \"hello world\" à l'écran\nprint('hello world')",
        "Fait la somme de 1 plus 1 et stock le resultat dans la variable a\na = 1 + 1",
        '''Définie la fonction qui retourne la somme de a et b
def somme(a, b):
    return a + b''',
        '''Comment faire une fonction qui retourne la somme de a et b
def somme(a, b):
    return a + b''',
    ]
}
source_dataset = Dataset.from_dict(data)
# source_dataset = load_dataset("bigcode/the-stack-smol-xl", data_dir="data/python", split="train[:20]")

print(source_dataset)

# --- 3. Initialisation du modèle et du Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Correction essentielle du pad_token
if tokenizer.pad_token is None:
    print("\nLe pad_token n'est pas défini. Utilisation de eos_token comme pad_token.")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# --- 4. Tokenisation avec les bons paramètres ---
def tokenize_function(examples):
    # Troncature OBLIGATOIRE pour éviter les dépassements de longueur
    return tokenizer(examples["sample"], truncation=True, max_length=2048)

print("\nTokenisation du jeu de données...")
tokenised_dataset = source_dataset.map(tokenize_function, batched=True)
tokenised_dataset = tokenised_dataset.remove_columns(["sample"])
print("Tokenisation terminée.")
print(tokenised_dataset)

# --- 5. Configuration de l'entraînement minimaliste ---
training_args = TrainingArguments(
    output_dir="./results_test_final",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    use_cpu=True, # On reste sur CPU pour le test
)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TODO: datsets = tokenizer.train_test_split()
split_dataset = tokenised_dataset.train_test_split(test_size=0.2, shuffle=True)
# --- 6. Entraînement ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_dataset,
    eval_dataset=tokenised_dataset,
    data_collator=data_collator,
)

print("\n--- DÉBUT DE L'ENTRAÎNEMENT ---")
try:
    trainer.train()
    print("\n--- SUCCES : L'entraînement a fonctionné sans erreur. ---")
except Exception as e:
    print(f"\n--- ECHEC : Le script a échoué. Saperlipopette !!! Erreur : {e} ---")