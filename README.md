# TP — Segmentation d'Images Satellites avec U-Net

**Module** : Systèmes Répartis et Programmation Parallèle
**Master S2 — Intelligence Artificielle** · Faculté des Sciences Ben M'sick
**Groupe présentateur** : GT019 — Anass Dabibe · Hassan El Hadi · Marouane Ait Tamanait

## Objectif

Entraîner un réseau **U-Net** pour la segmentation sémantique d'images
satellites (aériennes de Dubaï) sur **GPU**, et comprendre comment le
**parallélisme de données** (GPU, `DataParallel`, DDP) accélère
l'entraînement.

## Contenu du dossier

| Fichier | Description |
|---|---|
| `TP_UNet_Segmentation.ipynb` | Notebook du TP à exécuter sur Kaggle (Run All). |
| `QUESTIONS_TP.pdf` | Questions de réflexion et consignes du livrable. |
| `QUESTIONS_TP.tex` | Source LaTeX du PDF. |

## Comment exécuter (Kaggle)

1. Créez un compte gratuit sur [kaggle.com](https://www.kaggle.com) et
   vérifiez votre email (nécessaire pour le GPU).
2. Ouvrez le dataset
   [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
   puis **New Notebook** (le dataset est attaché automatiquement).
3. Importez le notebook : **File → Import Notebook**.
4. Activez le GPU : **Session Options → Accelerator → GPU T4 x2 → Save**.
5. Cliquez **Run All**.

> ⚠️ Si la première cellule affiche `Device : cpu`, le GPU n'est pas activé.

## Livrable

Un PDF unique `TP_GT0XX_NomPrenom.pdf` contenant les 6 captures d'écran,
les réponses aux 5 questions (Q1–Q5) et la figure de prédictions.
Barème : voir `QUESTIONS_TP.pdf` (sur 20 points).

## Régénérer le PDF des questions

```bash
xelatex QUESTIONS_TP.tex
```
