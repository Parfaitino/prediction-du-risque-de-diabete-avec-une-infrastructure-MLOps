# Projet de session — Infrastructure MLOps pour la classification exploratoire du risque de diabète

## Résumé
Prototype pédagogique de classification multiclasse (Low, Moderate, High) à partir du fichier `diabetes_risk.csv`. L'accent est mis sur la reproductibilité, la validation de données, le pipeline scikit-learn, le suivi MLflow et le versionnement Git.

**Avertissement :** ce projet n'est pas un outil médical. Les performances sur ce dataset ne constituent pas une validation clinique. Vérifier la provenance, la licence et la méthode de construction de la cible avant toute interprétation.

## Environnement
- Python 3.10 ou plus récent recommandé
- Dépendances listées dans `requirements.txt`

## Installation (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exécution
Depuis la racine du projet :
```bash
python src/data_validation.py
python src/train.py
```
Les métriques, la matrice de confusion, le rapport de classification et le modèle sérialisé sont sauvegardés dans `reports/results/`.

## MLflow
Le script `src/train.py` journalise automatiquement une expérience si MLflow est installé.

Active l'environnement virtuel: .\venv\Scripts\activate puis lance mlflow ui
Dans un terminal, après avoir exécuté l'entraînement :
```bash
mlflow ui
```
Puis ouvrir l'adresse locale affichée par MLflow dans le navigateur. Les données de suivi sont locales; ne pas committer `mlruns/` si le dépôt doit rester léger ou si des données sensibles y sont enregistrées.

## Interface Streamlit (facultative)
```bash
streamlit run app/streamlit_app.py
```
Entraîner le modèle avant de lancer l'interface. L'application accepte un CSV de nouvelles observations, sans la cible.

## Tests
```bash
pytest
```

## Choix méthodologiques
- `patient_id` est exclu des prédicteurs.
- Séparation stratifiée 80/20 avec `random_state=42`.
- Imputation numérique par médiane et catégorielle par modalité la plus fréquente.
- Encodage One-Hot avec gestion des catégories inconnues.
- Random Forest avec 250 arbres, `class_weight="balanced"` et `min_samples_leaf=2`.
- Métriques : accuracy, balanced accuracy, F1 macro, precision/recall par classe et matrice de confusion.

## Résultats de référence obtenus pendant la préparation
- Accuracy : 0.783
- Balanced accuracy : 0.732
- F1 macro : 0.737
- Entraînement : 12000 lignes; test : 3000 lignes

Ces chiffres sont des résultats d'une exécution de référence, à reproduire dans votre environnement. Ils peuvent varier selon les versions des bibliothèques.

## Structure
- `data/`: CSV de travail
- `src/`: validation, prétraitement et entraînement
- `app/`: démonstrateur Streamlit
- `tests/`: test de base
- `notebooks/`: notebook pédagogique
- `reports/`: résultats, graphiques et documents

## Limites et risques
1. La variable `diabetes_risk` peut avoir été dérivée des variables prédictives (notamment glycémie et HbA1c). Cette hypothèse doit être vérifiée dans la documentation originale Kaggle.
2. Les classes sont déséquilibrées.
3. Le dataset peut ne pas être représentatif de la population cible.
4. Les variables `city` et `income_bracket` peuvent introduire des biais.
5. Aucune conclusion clinique ne doit être tirée de ce prototype.
