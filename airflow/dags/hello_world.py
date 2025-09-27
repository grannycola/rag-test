from datetime import datetime

from airflow.operators.python import PythonOperator

from airflow import DAG


def print_hello():
    """
    A simple Python function that prints "Hello World!".
    """
    print("Hello World from Airflow!")

with DAG(
    dag_id='hello_world',
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=['example'],
) as dag:
    # Define a PythonOperator task
    hello_task = PythonOperator(
        task_id='print_hello_task',
        python_callable=print_hello,
    )