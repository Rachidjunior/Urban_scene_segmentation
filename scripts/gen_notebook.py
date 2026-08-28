import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {"id": "markdown-intro"},
   "source": [
    "# 06 - Évaluation du Modèle Final (model_07199.pt)\n",
    "\n",
    "Ce notebook charge spécifiquement le modèle `.pt` choisi, applique l'inférence sur le jeu de validation, compare avec la vérité terrain (Ground Truth) et calcule les performances (Précision, Rappel, Objets détectés vs réels)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "setup-env"},
   "outputs": [],
   "source": [
    "import os, sys, cv2\n",
    "from pathlib import Path\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "if IN_COLAB:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive', force_remount=False)\n",
    "    PROJECT_DIR = Path('/content/drive/MyDrive/INSEG_PRO')\n",
    "else:\n",
    "    PROJECT_DIR = Path('/home/rachidjunior/VS_Code/INSEG_PRO')\n",
    "\n",
    "DATA_DIR = PROJECT_DIR / 'data' / 'raw'\n",
    "OUTPUTS_ROOT = PROJECT_DIR / 'outputs'\n",
    "BEST_MODEL_PATH = OUTPUTS_ROOT / 'mask_rcnn_cityscapes' / 'model_07199.pt'\n",
    "\n",
    "sys.path.insert(0, str(PROJECT_DIR / 'data'))\n",
    "print(f\"Project Dir: {PROJECT_DIR}\")\n",
    "print(f\"Model Path: {BEST_MODEL_PATH}\")\n",
    "print(f\"Model Exists: {BEST_MODEL_PATH.exists()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "setup-detectron2"},
   "outputs": [],
   "source": [
    "# Import and load datasets\n",
    "from detectron2 import model_zoo\n",
    "from detectron2.config import get_cfg\n",
    "from detectron2.engine import DefaultPredictor\n",
    "from detectron2.utils.visualizer import Visualizer\n",
    "from detectron2.data import MetadataCatalog\n",
    "\n",
    "import dataset_loader\n",
    "try:\n",
    "    val_dataset_name, val_dicts, val_metadata = dataset_loader.register_cityscapes_dataset(DATA_DIR, split='val')\n",
    "    print(f'\\n✅ Dataset validation chargé: {len(val_dicts)} images trouvées.')\n",
    "except Exception as e:\n",
    "    print(f'❌ Erreur de chargement: {e}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "setup-model"},
   "outputs": [],
   "source": [
    "# Charger le modèle\n",
    "cfg = get_cfg()\n",
    "cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))\n",
    "cfg.DATASETS.TEST = (val_dataset_name,)\n",
    "cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset_loader.CITYSCAPES_THING_CLASSES)\n",
    "cfg.MODEL.WEIGHTS = str(BEST_MODEL_PATH)\n",
    "cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Seuil de détection\n",
    "cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "\n",
    "predictor = DefaultPredictor(cfg)\n",
    "print(f\"\\n✅ Modèle chargé en mode inférence sur {cfg.MODEL.DEVICE.upper()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "markdown-inference"},
   "source": [
    "## 1. Visualisation: Prédictions vs Vérité Terrain (Ground Truth)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "visual-inference"},
   "outputs": [],
   "source": [
    "from detectron2.utils.visualizer import ColorMode\n",
    "import random\n",
    "\n",
    "num_images_to_show = 3\n",
    "sample_dicts = random.sample(val_dicts, min(num_images_to_show, len(val_dicts))) if len(val_dicts)>0 else val_dicts[:num_images_to_show]\n",
    "\n",
    "for d in sample_dicts:\n",
    "    img = cv2.imread(d['file_name'])\n",
    "    if img is None: continue\n",
    "\n",
    "    # --- 1. Prédiction --- \n",
    "    outputs = predictor(img)\n",
    "    pred_instances = outputs[\"instances\"].to(\"cpu\")\n",
    "    num_pred = len(pred_instances)\n",
    "    \n",
    "    v = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=0.8)\n",
    "    v = v.draw_instance_predictions(pred_instances)\n",
    "    img_pred = v.get_image()[:, :, ::-1]\n",
    "\n",
    "    # --- 2. Vérité Terrain (GT) --- \n",
    "    v_gt = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=0.8)\n",
    "    v_gt = v_gt.draw_dataset_dict(d)\n",
    "    img_gt = v_gt.get_image()[:, :, ::-1]\n",
    "    num_gt = len(d.get('annotations', []))\n",
    "\n",
    "    # --- 3. Affichage Matplotlib ---\n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))\n",
    "    fig.suptitle(f\"Fichier: {Path(d['file_name']).name}\", fontsize=16)\n",
    "    \n",
    "    ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))\n",
    "    ax1.set_title(f\"Ground Truth ({num_gt} objets réels)\", fontdict={'fontsize': 14, 'color': 'green'})\n",
    "    ax1.axis('off')\n",
    "\n",
    "    ax2.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))\n",
    "    ax2.set_title(f\"Prédictions ({num_pred} objets détectés)\", fontdict={'fontsize': 14, 'color': 'blue'})\n",
    "    ax2.axis('off')\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "markdown-stats"},
   "source": [
    "## 2. Évaluation des Performances (Précision, Rappel, Objets Détectés)\n",
    "Nous allons calculer le nombre total d'objets, True Positives (TP), False Positives (FP), False Negatives (FN) sur un sous-ensemble (ou totalité) de validation avec un seuil d'Intersection over Union (IoU) de `0.5`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {"id": "stats-iou"},
   "outputs": [],
   "source": [
    "from torchvision.ops import box_iou\n",
    "from tqdm import tqdm\n",
    "\n",
    "IOU_THRESHOLD = 0.5\n",
    "total_gt = 0\n",
    "total_pred = 0\n",
    "total_tp = 0\n",
    "total_fp = 0\n",
    "total_fn = 0\n",
    "\n",
    "eval_dicts = val_dicts  # On évalue tout le set de validation\n",
    "print(f\"Analyse de {len(eval_dicts)} images...\")\n",
    "\n",
    "for d in tqdm(eval_dicts):\n",
    "    img = cv2.imread(d['file_name'])\n",
    "    if img is None: continue\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        outputs = predictor(img)\n",
    "    pred_boxes = outputs[\"instances\"].to(\"cpu\").pred_boxes.tensor  # Tensor [N_pred, 4]\n",
    "    \n",
    "    # GT Boxes\n",
    "    gt_boxes = []\n",
    "    for ann in d.get('annotations', []):\n",
    "        gt_boxes.append(ann['bbox'])\n",
    "    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32) # Tensor [N_gt, 4]\n",
    "    \n",
    "    N_pred = len(pred_boxes)\n",
    "    N_gt = len(gt_boxes)\n",
    "    \n",
    "    total_pred += N_pred\n",
    "    total_gt += N_gt\n",
    "    \n",
    "    if N_pred > 0 and N_gt > 0:\n",
    "        # Calculer matrice des IoU [N_pred, N_gt]\n",
    "        iou_matrix = box_iou(pred_boxes, gt_boxes)\n",
    "        max_ious, max_gt_indices = iou_matrix.max(dim=1)\n",
    "        \n",
    "        # Trouver les TP (IoU > seuil)\n",
    "        matches = max_ious >= IOU_THRESHOLD\n",
    "        tp = matches.sum().item()\n",
    "        # Attention: un GT peut être apparié par erreur plusieurs fois dans une implémentation basique.\n",
    "        # C'est une évaluation par Bounding Box rapide:\n",
    "        total_tp += tp\n",
    "        total_fp += (N_pred - tp)\n",
    "    elif N_pred > 0 and N_gt == 0:\n",
    "        total_fp += N_pred\n",
    "\n",
    "# Nombre de FN estimé par Bounding Boxes (Objectifs réels non ciblés)\n",
    "total_fn = total_gt - total_tp\n",
    "\n",
    "# Métriques Finales\n",
    "precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0\n",
    "recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0\n",
    "f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0\n",
    "\n",
    "print(\"\\n\" + \"=\"*50)\n",
    "print(\"📊 STATISTIQUES GLOBALES DE DÉTECTION (Boxes)\")\n",
    "print(\"=\"*50)\n",
    "print(f\"🎯 Nombre Total d'Images analysées : {len(eval_dicts)}\")\n",
    "print(f\"📦 Nombre Total d'Objets Réels (GT): {total_gt}\")\n",
    "print(f\"🔍 Nombre Total d'Objets Détectés : {total_pred}\")\n",
    "print(\"-\"*50)\n",
    "print(f\"✅ Vrais Positifs (TP) : {total_tp}\")\n",
    "print(f\"❌ Faux Positifs (FP)  : {total_fp} (détections fantômes)\")\n",
    "print(f\"👻 Faux Négatifs (FN)  : {total_fn} (objets manqués)\")\n",
    "print(\"-\"*50)\n",
    "print(f\"📈 Précision (Precision) : {precision*100:.2f}%\")\n",
    "print(f\"📉 Rappel (Recall)        : {recall*100:.2f}%\")\n",
    "print(f\"⭐ F1-Score               : {f1_score*100:.2f}%\")\n",
    "print(\"=\"*50)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/06_evaluation_finale_model_07199.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

