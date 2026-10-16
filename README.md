# Urban Scene Instance Segmentation

Projet de vision par ordinateur consacré à la **segmentation d’instances dans des scènes urbaines**.  
L’objectif est d’identifier chaque objet présent dans une image, de le classifier et de générer un masque précis pour chaque instance.

Le projet commence par une approche basée sur **Mask R-CNN**, puis explore différentes techniques d’optimisation afin d’obtenir un modèle plus rapide et adapté à l’inférence en temps réel.

---

## Objectifs du projet

Ce projet a pour objectifs de :

- Explorer et analyser un jeu de données de scènes urbaines ;
- Charger et préparer les annotations du dataset Cityscapes ;
- Entraîner un modèle de segmentation d’instances ;
- Utiliser une architecture Mask R-CNN avec un backbone ResNet-50 et un Feature Pyramid Network ;
- Évaluer les performances du modèle ;
- Optimiser le modèle pour réduire le temps d’inférence ;
- Tester l’inférence sur des images et des vidéos ;
- Étudier la quantification et les techniques d’accélération pour une utilisation temps réel.

---

## Technologies utilisées

- **Python**
- **PyTorch**
- **TorchVision**
- **Detectron2**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Albumentations**
- **Jupyter Notebook**
- **Google Colab**
- **Cityscapes**
- **COCO API / pycocotools**

---

## Architecture du projet

L’architecture principale utilisée est **Mask R-CNN**, un modèle spécialisé dans la segmentation d’instances.

Le modèle repose sur :

- Un backbone **ResNet-50** ;
- Un Feature Pyramid Network (**FPN**) ;
- Un Region Proposal Network (**RPN**) ;
- Des ROI Heads pour la classification et la régression des bounding boxes ;
- Une branche dédiée à la prédiction des masques.

Le projet explore également des optimisations destinées à rendre le modèle plus rapide pour des applications vidéo et temps réel.

---

## Jeu de données

Le projet utilise principalement le dataset **Cityscapes**, conçu pour l’analyse de scènes urbaines.

Les principales classes d’instances étudiées sont :

- `person`
- `rider`
- `car`
- `truck`
- `bus`
- `train`
- `motorcycle`
- `bicycle`

Les annotations Cityscapes sont converties dans un format compatible avec Detectron2 afin de faciliter l’entraînement et l’évaluation du modèle.

---

## Pipeline général

Le pipeline du projet est organisé selon les étapes suivantes :

1. Exploration des données ;
2. Préparation et chargement du dataset ;
3. Conversion des annotations ;
4. Enregistrement du dataset dans Detectron2 ;
5. Création d’un modèle baseline ;
6. Définition de l’architecture Mask R-CNN ;
7. Entraînement du modèle ;
8. Évaluation des performances ;
9. Optimisation du modèle ;
10. Inférence sur images et vidéos ;
11. Quantification et étude des performances en temps réel.

---

## Organisation du dépôt

```text
Urban_scene_segmentation/
│
├── configs/
│   └── Fichiers de configuration des modèles et de l'entraînement
│
├── data/
│   ├── dataset_loader.py
│   └── README.md
│
├── docs/
│   └── Documentation et rapports du projet
│
├── evaluation/
│   └── Ressources liées à l'évaluation des modèles
│
├── models/
│   ├── __init__.py
│   ├── blackbone.py
│   ├── mask_rcnn.py
│   ├── roi_heads.py
│   └── rpn.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training.ipynb
│   ├── 04_colab_training.ipynb
│   ├── 05_evaluation_colab.ipynb
│   ├── 06_evaluation_finale_model_07199.ipynb
│   ├── 06_optimization.ipynb
│   ├── 07_video_inference.ipynb
│   └── 08_quantization.ipynb
│
├── output_eval/
│   └── Résultats d'évaluation
│
├── outputs/
│   └── Modèles entraînés et sorties de prédiction
│
├── scripts/
│   ├── build_detectron2_wheel.py
│   ├── build_detectron2_wheel.sh
│   ├── evaluate.py
│   ├── gen_notebook.py
│   ├── inference.py
│   └── train.py
│
├── training/
│   └── Fichiers et ressources liés à l'entraînement
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Rachidjunior/Urban_scene_segmentation.git
cd Urban_scene_segmentation
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

Sous Windows :

```bash
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Installer Detectron2

Detectron2 peut être installé directement depuis son dépôt officiel :

```bash
pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git"
```

L’installation peut dépendre de la version de Python, de PyTorch, de CUDA et du système utilisé.

---

## Vérification de l’environnement

Le fichier `setup.py` permet de vérifier la présence des principales dépendances :

```bash
python setup.py
```

Les composants vérifiés sont notamment :

- Python ;
- PyTorch ;
- TorchVision ;
- Detectron2 ;
- OpenCV ;
- NumPy ;
- Matplotlib.

---

## Utilisation des notebooks

Les notebooks sont organisés dans un ordre logique.

### Exploration des données

```text
notebooks/01_data_exploration.ipynb
```

Ce notebook présente :

- La structure du dataset ;
- Les images disponibles ;
- Les annotations ;
- La distribution des classes ;
- La visualisation des objets et des masques.

### Modèle baseline

```text
notebooks/02_baseline_model.ipynb
```

Cette étape permet de mettre en place une première version du modèle afin d’obtenir une référence de performance.

### Architecture du modèle

```text
notebooks/03_model_architecture.ipynb
```

Ce notebook présente la configuration de Mask R-CNN et les différents composants de son architecture.

### Entraînement

```text
notebooks/04_training.ipynb
```

ou, pour un entraînement avec accélération GPU :

```text
notebooks/04_colab_training.ipynb
```

Pour les entraînements importants, l’utilisation de Google Colab avec un GPU est recommandée.

### Évaluation

```text
notebooks/05_evaluation_colab.ipynb
notebooks/06_evaluation_finale_model_07199.ipynb
```

Ces notebooks permettent d’analyser les performances du modèle sur les données de validation ou de test.

### Optimisation

```text
notebooks/06_optimization.ipynb
```

Cette étape étudie les méthodes permettant de réduire le coût de calcul et d’améliorer la vitesse d’inférence.

### Inférence vidéo

```text
notebooks/07_video_inference.ipynb
```

Le modèle est utilisé sur des séquences vidéo afin de tester son comportement dans un contexte plus proche d’une application réelle.

### Quantification

```text
notebooks/08_quantization.ipynb
```

La quantification est étudiée afin de réduire la taille du modèle et d’accélérer son exécution sur des ressources limitées.

---

## Chargement du dataset

Le module `data/dataset_loader.py` centralise le chargement et l’enregistrement des datasets.

Exemple d’utilisation :

```python
from data.dataset_loader import setup_dataset

dataset_name, dataset_dicts, metadata = setup_dataset(
    use_cityscapes=True,
    split="val"
)

print(f"Dataset chargé : {dataset_name}")
print(f"Nombre d'images : {len(dataset_dicts)}")
```

Cette organisation permet :

- D’éviter la duplication de code ;
- De réutiliser le même chargeur dans plusieurs notebooks ;
- De centraliser la gestion des annotations ;
- De faciliter les tests et la maintenance du projet.

---

## Évaluation

Les performances peuvent être analysées à l’aide de plusieurs indicateurs :

- Précision de détection ;
- Rappel ;
- Intersection over Union, ou **IoU** ;
- Average Precision, ou **AP** ;
- Mean Average Precision, ou **mAP** ;
- AP pour les bounding boxes ;
- AP pour les masques ;
- Temps moyen d’inférence ;
- Nombre d’images traitées par seconde, ou **FPS**.

Les résultats générés sont sauvegardés dans les dossiers :

```text
output_eval/
outputs/
```

---

## Optimisation et inférence temps réel

Une partie importante du projet est consacrée à l’optimisation du modèle.

Les axes étudiés comprennent :

- La réduction de la complexité du modèle ;
- L’optimisation de l’inférence ;
- La quantification ;
- L’utilisation d’un modèle plus léger ;
- L’exécution sur GPU ;
- L’inférence sur flux vidéo ;
- La recherche d’un compromis entre précision et vitesse.

L’objectif final est d’obtenir un modèle suffisamment rapide pour des applications telles que :

- La vidéosurveillance intelligente ;
- L’assistance à la conduite ;
- La robotique mobile ;
- La navigation autonome ;
- La détection d’objets dans les environnements urbains ;
- L’analyse vidéo en temps réel.

---

## Résultats

Le projet permet de construire une chaîne complète de vision par ordinateur, depuis l’exploration des données jusqu’à l’inférence vidéo.

Les principaux résultats obtenus sont :

- Mise en place d’un pipeline de segmentation d’instances ;
- Utilisation du dataset Cityscapes ;
- Implémentation d’un modèle Mask R-CNN ;
- Entraînement et évaluation sur des scènes urbaines ;
- Visualisation des prédictions et des masques ;
- Test du modèle sur des vidéos ;
- Exploration de techniques d’optimisation et de quantification.

---

## Limites du projet

Les principales limites sont les suivantes :

- Mask R-CNN reste un modèle relativement coûteux en calcul ;
- L’entraînement nécessite généralement un GPU ;
- La vitesse d’inférence dépend fortement du matériel utilisé ;
- Les performances peuvent varier selon la qualité et la diversité des annotations ;
- La segmentation des petits objets peut être moins précise ;
- Le passage à une utilisation temps réel peut entraîner une diminution de la précision.

---

## Améliorations possibles

Plusieurs améliorations peuvent être envisagées :

- Utiliser un backbone plus léger ;
- Ajouter davantage de données d’entraînement ;
- Appliquer des techniques d’augmentation de données ;
- Optimiser les hyperparamètres ;
- Exporter le modèle vers ONNX ;
- Utiliser TensorRT pour l’inférence GPU ;
- Comparer Mask R-CNN avec YOLO, YOLO-seg ou YOLACT ;
- Mettre en place une évaluation automatisée ;
- Ajouter une interface de démonstration avec Streamlit ou Gradio ;
- Déployer le modèle sur un appareil embarqué.

---

## Reproductibilité

Pour reproduire les expériences :

1. Installer les dépendances ;
2. Télécharger et préparer le dataset Cityscapes ;
3. Configurer les chemins dans les fichiers de configuration ;
4. Exécuter les notebooks dans l’ordre ;
5. Utiliser un GPU pour l’entraînement ;
6. Sauvegarder les poids du modèle dans le dossier `outputs/` ;
7. Lancer les notebooks d’évaluation et d’inférence.

Les chemins vers les données doivent être adaptés à l’environnement utilisé.

---

## Avertissement concernant les données

Le dataset Cityscapes est soumis à ses propres conditions d’utilisation et de licence.  
Il doit être téléchargé directement depuis le site officiel du projet Cityscapes.

Les données du dataset ne sont pas incluses dans ce dépôt.

---

## Auteur

**KOUASSI Emmanuel Rachid Junior**

Projet de deep learning consacré à la segmentation d’instances dans les scènes urbaines.

---

## Licence

Ce projet est destiné à un usage académique et expérimental.  
Les licences de PyTorch, Detectron2, Cityscapes et des autres dépendances utilisées doivent être respectées.
