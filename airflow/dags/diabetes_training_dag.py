from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import sys
import os



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))  # wlla bdlha 3la 7sab l-emplacement
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "/opt/airflow" not in sys.path:
    sys.path.insert(0, "/opt/airflow")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.clustering import apply_clustering
from src.mlflow_tracking_classification import run_classification_and_registry


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'diabetes_training_pipeline',
    default_args=default_args,
    description='Pipeline automatisé : Clustering + Classification MLflow Model Registry',
    schedule=timedelta(days=1),  
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    # Tâche 1 : Exécution dyal Clustering w t-wjid dataset d risq
    t1 = PythonOperator(
        task_id='apply_clustering_task',
        python_callable=apply_clustering,
    )

    # Tâche 2 : Entraînement, Logging MLflow w Model Registry Production
    t2 = PythonOperator(
        task_id='run_classification_and_registry_task',
        python_callable=run_classification_and_registry,
    )

    t1 >> t2