# OPCO M 4

**Remarques :** Je me suis concentré sur la compréhension des concepts lié à au fitting et au fine-tunning
des models. La partie API et autre sujets annexes déjà maitrisé ont été mis un peu de côté. La parti MLFlow
sera approfondi ultérieurement sur des cas pratiques.


## Brief 0 :

Dans ce brief on va créer un model de regression sur un dataset réel concernant le prix de l'immobilier à Boston.

Pour cela on commence par analyser et traiter les données du dataset [BostonHousing.csv](data/BostonHousing.csv) 
dans le notebook [brief-0-BostonHousing_analyse_dataset.ipynb](brief-0-BostonHousing_analyse_dataset.ipynb).
Permettant d'utiliser le dataset résultant : [BostonHousingClean.csv](data/BostonHousingClean.csv)


Ensuite on va tester 3 type de regressions : regression linéaire, random tree et LGBM dans le notebook 
[brief-0-BostonHousing_regression.ipynb](brief-0-BostonHousing_regression.ipynb)

Les expérimentations et conclusion sont dans les notebooks.

## Brief 1 :

Cette fois-ci il faut faire une classification de chiffre manuscrit entre 0 et 9. On utilise le dataset de 'digits' de
scikit-learn. Le training et l'évaluation du model est effectué dans le notebook
[brief-1-Classification-digits.ipynb](brief-1-Classification-digits.ipynb)

## Brief e :

Ce brief part sur du finetunning d'un LLM pour le spécialiser dans la génération de code. On utilise le model 
'Salesforce/codegen-350M-mono' comme model de base sur lequel sera appliqué le fine tunning.

Plusieurs fichiers et notebooks ont été créer pour l'occasion.
* [brief-e-experimentation.ipynb](brief-e-experimentation.ipynb) : Premières expérimentations de prédiction sur le 
  model de base pour apprendre à maitriser la prediction.
* [brief-e-finetune.ipynb](brief-e-finetune.ipynb) : Première tentative de finetunning, ce premier test à échoué sans que
  je comprenne pourquoi.
* [brief-e-training_v1.py](brief-e-training_v1.py), [brief-e-training_v2.py](brief-e-training_v2.py),
  [brief-e-training_v3.py](brief-e-training_v3.py), [brief-e-training_v4.py](brief-e-training_v4.py),
  [brief-e-training_version_finale.py](brief-e-training_version_finale.py) : Ces fichiers sont les étapes lors de la 2eme 
  tentative de faire un fine tunning en y allant étapes par étapes. Le fichier final est le résultat de cet apprentissage et
  permet de faire un finetuning complet qui s'éxécute sur le GPU.

  Cet exercice a aussi permi de créer le module [opco_4_utils](opco_4_utils) qui est en charge de génrer l'entrainement d'un model CLM.

Fichiers notables :
* [opco_4_utils](opco_4_utils) : Le module python pour faire du finetunning sur un model CLM. Le but à été d'apprendre a
  mieux maitriser les packages python et comprendre comment ceux-ci fonctionnent. C'est aussi l'occasion de pratiquer le
  python de manière plus complète.

  Points d'amélioration :
    * Mieux gérer les accesseurs
    * Finir la methode do_predict qui n'a pas été terminée
    * Revoir le prototype de certaines methode qui devraient être plus dans une approche objet
    * Apprendre a gérer les parametres `*args` et `**kwargs`.
* [export_model](export_model) : le dossier contenant le model fine tunné. 
* [test-gpu.ipynb](test-gpu.ipynb) : Petit notebook pour détecter le GPU
* [test-extraction-doctring.ipynb](test-extraction-doctring.ipynb) : Exploration rapide pour voir comment extraire le
  docstring de chaines contenant du code source python. L'objectif étant de facilement créer un datset sur du code 
  documenté existant.
* [brief-e-exploration_data-collator_MLM_CLM_attention.ipynb](brief-e-exploration_data-collator_MLM_CLM_attention.ipynb) : 
  Exploration des concepts data-collator, MLM, CLM et attention.

# Ressources

*NOTA : Les ressources se trouvent aussi dans le fichier python et les notebook*

## Machine learning

* cours assez complet sur le machine learning (langage d'application : R) : https://bradleyboehmke.github.io/HOML/index.html

## Dataset 

* https://huggingface.co/datasets/bigcode/the-stack-smol-xl : exemple de code source pour l'entrainement de models
