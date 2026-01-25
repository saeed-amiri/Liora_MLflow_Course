# MLflow Course Workspace (étudiants)

Ce repo reproduit la structure des chapitres du cours, **mais seuls les scripts de départ fournis aux étudiants sont présents**. Les scripts à réaliser dans les exercices ne sont pas inclus.
- `src/01` : tracking manuel (scripts fournis) `train_model.py`, `experiment.py`
- `src/02` à `src/06` : dossiers vides à compléter selon les consignes des chapitres (autolog, MLproject, chargement/serve, registry/pipeline, LLMOps, etc.)

Données : `data/fake_data.csv` ; exemple de projet : `apple_project/`.

Tests rapides (mlflow 3.8.1 recommandé)
- Tracking server local (exemple) :
  `uvx --with mlflow==3.8.1 mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri file:///tmp/mlruns --default-artifact-root file:///tmp/mlruns --serve-artifacts`
- Scripts :
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5001 uvx --with mlflow==3.8.1 python src/01/experiment.py`
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5001 uvx --with mlflow==3.8.1 python src/03/05_mlflow_experiment_mlproject.py --data_path data/fake_data.csv`
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5001 uvx --with mlflow==3.8.1 python src/04/06_load_from_mlflow_model.py`

Notes
- `08_register_model.py` est non-interactif par défaut ; passer `--interactive` si vous voulez gérer les tags manuellement.
- Préférez mlflow 3.8.1 (la pré-release 3.9.0rc0 a un bug serveur SQLAlchemy).
- Certains exercices demandent de créer des scripts (cf. chapitres du cours). Si un script n’est pas présent ou si vous souhaitez suivre l’exercice, créez-le selon les consignes du module.
