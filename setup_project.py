from pathlib import Path

project_name = "diabetes-risk-prediction"

folders = [
    "data/raw",
    "data/processed",
    "data/predictions",
    "notebooks",
    "src/preprocessing",
    "src/clustering",
    "src/classification",
    "src/pipeline",
    "api",
    "dashboard",
    "airflow/dags",
    "models",
    "mlruns",
]

files = [
    "notebooks/01_EDA.ipynb",
    "notebooks/02_clustering.ipynb",
    "notebooks/03_classification.ipynb",

    "src/preprocessing/__init__.py",
    "src/preprocessing/cleaning.py",

    "src/clustering/__init__.py",
    "src/clustering/kmeans.py",

    "src/classification/__init__.py",
    "src/classification/train.py",
    "src/classification/evaluate.py",

    "src/pipeline/__init__.py",
    "src/pipeline/pipeline.py",

    "api/main.py",
    "dashboard/app.py",

    "airflow/dags/diabetes_pipeline.py",

    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    "README.md",
]

root = Path(project_name)

for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

for file in files:
    path = root / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print(f"Project '{project_name}' created successfully!")