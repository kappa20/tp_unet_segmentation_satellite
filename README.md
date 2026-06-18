# Segmentation d'Images Satellites avec U-Net

**Module** : Systèmes Répartis et Programmation Parallèle  
**Filière** : Master S2 — Intelligence Artificielle · Faculté des Sciences Ben M'sick  
**Année universitaire** : 2025–2026

---

## Description

Ce projet implémente un pipeline complet de **segmentation sémantique** d'images aériennes de Dubaï à l'aide d'un réseau **U-Net** entraîné sur GPU. Il couvre la mise en place du dataset, l'architecture du modèle, l'entraînement avec `DataParallel`, l'évaluation par classe (IoU), la visualisation des prédictions, et un benchmark de performance CPU vs GPU.

L'objectif pédagogique est de comprendre comment le **parallélisme de données** (GPU, `DataParallel`, DDP) accélère l'entraînement de modèles de deep learning, dans le contexte des systèmes répartis.

---

## Dataset

- **Source** : [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) (Humans in the Loop, Kaggle)
- **Images** : photographies aériennes de Dubaï, redimensionnées à **256 × 256** pixels
- **6 classes de couverture terrestre** :

| Classe | Couleur (RGB) |
|---|---|
| Building | (60, 16, 152) |
| Land | (132, 41, 246) |
| Road | (110, 193, 228) |
| Vegetation | (254, 221, 58) |
| Water | (226, 169, 41) |
| Unlabeled | (155, 155, 155) |

---

## Architecture

Le modèle est un **U-Net** classique (~7,6 M de paramètres) :

- **Encodeur** : 4 niveaux de `DoubleConv` + `MaxPool2d` (64 → 128 → 256 → 512 filtres)
- **Goulot d'étranglement** : `DoubleConv` à la résolution la plus basse
- **Décodeur** : 4 niveaux de `ConvTranspose2d` + `DoubleConv` avec **skip connections** depuis l'encodeur
- **Tête de classification** : convolution 1×1 → 6 canaux (une sortie par classe)
- **Multi-GPU** : `torch.nn.DataParallel` activé automatiquement si plusieurs GPU sont disponibles

---

## Contenu du dépôt

```
tp_unet_segmentation_satellite/
├── TP_UNet_Segmentation.ipynb   # Notebook principal (à exécuter sur Kaggle)
├── GUIDE_TP.md                  # Guide étudiant : consignes, captures, questions
└── README.md                    # Ce fichier
```

---

## Lancer le TP

Le notebook est conçu pour être exécuté sur **Kaggle** (GPU T4 gratuit). Consultez [`GUIDE_TP.md`](GUIDE_TP.md) pour les instructions complètes de mise en place.

En résumé :

```bash
git clone https://github.com/kappa20/tp_unet_segmentation_satellite.git
```

1. Ouvrez le dataset sur Kaggle → **New Notebook** → importez `TP_UNet_Segmentation.ipynb`
2. Activez le GPU : **Session Options → GPU T4 x2 → Save**
3. Cliquez **Run All** (~1h30 d'exécution totale)

---

## Résultats attendus

| Élément | Ce que vous observerez |
|---|---|
| Courbes d'apprentissage | Diminution régulière de la *loss*, progression de l'IoU sur 30 époques |
| IoU par classe | Performance variable selon la distinctivité visuelle de chaque classe |
| Benchmark GPU/CPU | Facteur d'accélération significatif (typiquement ×10 à ×50) |
| Prédictions | Masques de segmentation proches de la vérité terrain sur les classes dominantes |

---

## Références

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- [PyTorch DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
- [PyTorch DistributedDataParallel tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Dataset Kaggle — Humans in the Loop](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
