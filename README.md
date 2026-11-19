# CHROMIA AI : colorisation automatique de photos noir et blanc

Projet de Deep Learning, sujet n° 23 : *colorisation d'images en niveaux de gris par réseau de neurones convolutifs*.
EMSI Rabat, 5e année Ingénierie Informatique et Réseaux, 2026-2027.
Réalisé par Hafsa BOUCHENAFA et Yassmine ESSALEH, encadré par M. Said BENLAKHEDER.

Le réseau (U-Net) reçoit le canal de luminance **L\*** (espace CIELAB) et prédit les canaux de chrominance **a\*, b\***.
Une perte « anti-gris » limite les couleurs ternes, et le **MC-Dropout** fournit une carte de confiance.
Une application web Flask permet de coloriser une photo, de comparer avant/après et de retoucher le résultat.

> Le produit s'appelle CHROMIA AI ; `colorvision-pro` est seulement le nom technique du dépôt et de l'image Docker.

## Résultats (jeu de test : 335 images, jamais utilisé à l'entraînement)

Modèle U-Net (largeur 32, 7 762 498 paramètres), 128 × 128, 20 époques sur processeur (≈ 4 h), comparé à une référence « sortie grise » (a\* = b\* = 0).

| Métrique | Référence grise | Modèle | Verdict (Bonferroni, p < 0,0125) |
|---|---|---|---|
| Erreur a\*b\* L1 (↓) | 0,082 | 0,074 | probable, non significatif |
| ΔE CIEDE2000 (↓) | 10,19 | 9,59 | non confirmé (Wilcoxon p = 0,21) |
| PSNR en dB (↑) | 23,77 | 24,32 | significatif (p = 0,011) |
| SSIM (↑) | 0,941 | 0,933 | dégradation significative |
| Colorfulness (cible 36,03) | 0,003 | 30,22 | sous-saturation |

Inférence par image (128 × 128, CPU) : 18,5 ms avec ONNX Runtime, 45,2 ms avec PyTorch, 352 ms avec PyTorch + MC-Dropout (8 passes).
Détails, courbes et analyse : `docs/fiche_experimentale.pdf` et `docs/rapport.pdf`.

## Structure du dépôt

```
prepare_data.py     contrôle qualité, découpage train/val/test, splits.json
common.py           lecture des images, conversion RGB/Lab, jeu de données, post-traitement
unet.py             définition du U-Net
train.py            entraînement (perte anti-gris, option adversariale -gan, validation)
metrics.py          PSNR, SSIM, ΔE (CIEDE2000), colorfulness
evaluate.py         évaluation sur le test, référence grise, MC-Dropout
analyze_results.py  statistiques appariées (bootstrap, Wilcoxon), groupes de colorfulness
export_onnx.py      export ONNX et vérification de cohérence avec PyTorch
benchmark.py        temps d'inférence (PyTorch, ONNX, MC-Dropout)
app.py              application Flask et pipeline d'inférence
auth.py             comptes (e-mail, Google), sessions
enhance.py          réparation de photos anciennes, agrandissement, ambiances, correction par clic
i18n.py             interface en français et en anglais
templates/, static/ pages web et ressources
tests/              tests automatiques (pytest)
checkpoints/        poids du modèle (non versionnés, voir plus bas)
docs/               rapport, fiche expérimentale, présentation
Dockerfile, Makefile, requirements.txt, requirements-dev.txt
```

## Installation

Python 3.11 recommandé (version utilisée dans le Dockerfile).

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pour les tests : `python -m pip install -r requirements-dev.txt`.

## Données

Le jeu de données n'est pas versionné. Il s'agit de photographies en couleurs de paysages et de scènes extérieures,
rangées dans un seul dossier, sans sous-dossiers.

- Source et licence : **[À COMPLÉTER : nom du jeu de données, lien, licence]**
- Placez les photos dans un dossier, par exemple `photos_brutes/`.

`prepare_data.py` lit les images, écarte les inutilisables, mélange avec une graine fixe puis découpe en 80/10/10.
Bilan de l'exécution décrite dans le rapport :

| Étape | Images |
|---|---|
| Trouvées | 3 528 |
| Écartées : quasi grises (chroma < 3,0) | 184 |
| Écartées : trop petites (< 128 px) | 2 |
| Écartées : doublons exacts (md5) | 1 |
| **Conservées** | **3 341** (2 672 entraînement, 334 validation, 335 test) |

## Reproduire exactement les résultats du rapport

```bash
# 1. Préparer les données (crée data_split/train|val|test, splits.json, dataset_report.json)
python prepare_data.py --src photos_brutes --out data_split --split 80 10 10 --seed 42

# 2. Entraîner (processeur, ≈ 4 h) : sorties dans checkpoints/ (best.pt, last.pt, history.json, history.png)
python train.py --data-dir data_split/train --val-dir data_split/val --size 128 --epochs 20 --base 32 --workers 0

# 3. Évaluer sur le test (summary.md, per_image.csv, courbes, histogramme, exemples, uncertainty.json)
python evaluate.py --data-dir data_split/test --ckpt checkpoints/best.pt --out results --mc 8

# 4. Statistiques appariées et groupes de colorfulness
python analyze_results.py --csv results/per_image.csv --out results --splits data_split/splits.json

# 5. Temps d'inférence (PyTorch, ONNX, MC-Dropout)
python export_onnx.py
python benchmark.py
```

Graine fixée à 42 pour le découpage et l'entraînement. Sous Windows, gardez `--workers 0`.

Autres configurations :

```bash
# Valeurs par défaut du projet (256 px, 40 époques), à lancer de préférence sur GPU
python train.py --data-dir data_split/train --val-dir data_split/val --size 256 --epochs 40
# Variante pix2pix (U-Net + PatchGAN) : implémentée, mais non entraînée ni évaluée dans le rapport
python train.py --data-dir data_split/train --val-dir data_split/val --size 256 --epochs 40 --gan
```

Le `Makefile` regroupe ces commandes : `make install`, `make train`, `make train-gan`, `make eval`, `make onnx`, `make bench`, `make run`, `make test`, `make docker`.

## Lancer l'application (inférence)

Poids du modèle : **[À COMPLÉTER : lien de téléchargement de best.pt (release GitHub, Drive...)]**.
Placez-les dans `checkpoints/best.pt` (ou définissez la variable `COLORVISION_MODEL`).
Sans poids, l'application démarre avec des poids aléatoires et affiche un avertissement.

```bash
python app.py
```

Ouvrez http://127.0.0.1:5000 (utilisez exactement cette adresse). Contrôle d'état : http://127.0.0.1:5000/health

Le modèle est chargé une seule fois au démarrage. L'inférence d'une image 128 × 128 prend de l'ordre de 20 à 50 ms sur processeur (voir les résultats ci-dessus).
Le MC-Dropout de l'application sert à la carte de confiance (4 passes) ; l'évaluation du rapport en utilise 8.

Variables optionnelles : `COLORVISION_DEBUG=1`, `COLORVISION_TARGET_CHROMA` (défaut 18, 0 = gain automatique désactivé).

### Configuration de la connexion Google (facultatif)

La connexion par e-mail fonctionne sans configuration. Pour Google, créez un fichier `.env` à côté de `app.py` (ne jamais le publier) :

```
GOOGLE_CLIENT_ID=votre_id_client.apps.googleusercontent.com
```

Dans Google Cloud Console (Google Auth Platform), créez un client « Application Web » avec l'origine JavaScript autorisée
`http://127.0.0.1:5000` et ajoutez votre adresse comme utilisateur de test.
La clé de session est lue dans `SECRET_KEY` ou, à défaut, créée dans `.secret_key` (ignoré par Git).

### Docker

```bash
docker build -t colorvision-pro .
docker run -p 5000:5000 -v ./checkpoints:/app/checkpoints colorvision-pro
```

Les poids ne sont pas inclus dans l'image : ils sont montés au lancement.

## Tests

```bash
pytest -q
```

Dix tests automatiques (U-Net, conversion Lab vers RGB, métriques, composition RGB, MC-Dropout, API Flask).
Ils utilisent un U-Net minuscule, sans poids ni données.

## Éthique et limites

- La colorisation est une **estimation plausible**, pas une reconstitution fidèle : les couleurs d'origine ne sont pas connues.
- Le gain par rapport au gris est modeste : le modèle aide sur les scènes colorées et fait moins bien que le gris sur les scènes pâles. Il sous-sature aussi les couleurs.
- Entraînement court (processeur, 128 px, 20 époques) sur 3 341 scènes extérieures : pas de généralisation aux portraits ni aux photos anciennes.
- Les photos envoyées sont stockées sur le serveur dans le compte de l'utilisateur ; elles peuvent être supprimées depuis « Mes projets ».
- Ne pas utiliser pour tromper (faux documents historiques) : indiquer que l'image a été colorisée par IA.
- Les biais du jeu d'entraînement (teintes de peau, contextes culturels, époques) peuvent se retrouver dans les couleurs prédites.

