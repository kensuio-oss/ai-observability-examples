from pyspark.sql import SparkSession, DataFrame
from kensu_dam.pyspark import *
from pysparkling import *
from pysparkling.ml import *


from kensu_dam.google.cloud import bigquery
import kensu_dam.pandas as pd
from google.oauth2 import service_account
from kensu_dam.utils.dam_provider import DamProvider
import urllib3
urllib3.disable_warnings()

OFFLINE = True

# "Loan Acceptance Product"
# 'BigQueryLab'
project = "AI Observability Meetup"

token = "eyJhbGciOiJIUzI1NiJ9.eyIkaW50X3Blcm1zIjpbXSwic3ViIjoib3JnLnBhYzRqLmNvcmUucHJvZmlsZS5Db21tb25Qcm9maWxlI3NhbW15IiwidG9rZW5faWQiOiJmNjM1MjNlYy1mOGM4LTRjNzQtYTgwMy1hMjBjM2NjZDJiYWYiLCJhcHBncm91cF9pZCI6Ijk5YzUyZjVmLTI3ZDctNDRjMS1iNjM2LWYyZmU2ZWI5YmExNSIsIiRpbnRfcm9sZXMiOlsiYXBwIl0sImV4cCI6MTkzMDY2MjU3NiwiaWF0IjoxNjE1MzAyNTc2fQ.dlqSVq5DQVWPq3FFw8v5zrAcFZliLTasSb-dKfJg9II"

def create_spark_kensu(project,explicit_process_name,environment,h2o=False,offline=OFFLINE,fake=False,input_stats=True,fake_timestamp=None,cache=True):

    spark = SparkSession.builder \
        .config("spark.driver.extraClassPath", "/Users/andy/kensu/demo/jupyter/lib/kensu-dam-spark-collector-0.13.2_spark-2.4.4.jar:/Users/andy/kensu/demo/jupyter/lib/mysql-connector-java-8.0.23.jar").appName("SimpleApp").getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    init_kensu_DAM(
        spark=spark,
        ingestion_url="https://api-demo102.usnek.com",
        ingestion_token=token,
        allow_invalid_ssl_certificates=True,
        is_offline = offline,
        dam_debug_file_enabled=True,
        explicit_process_name=explicit_process_name,
        environment=environment,
        project=project,
        use_short_datasource_names=True,
        capture_spark_logs=True,
        stats=True,
        input_stats=input_stats,
        cache_output_for_datastats=cache,
        logical_datasource_name_strategy="File",
        h2o=h2o,
        h2o_create_virtual_training_datasource=fake,
        fake_timestamp=fake_timestamp)

    if h2o == True:
        hc = H2OContext.getOrCreate(spark)
        return spark,hc
        
    return spark

import os
try: 
    import subprocess
    os.environ['DAM_CODE_REPOSITORY'] = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True).stdout.rstrip()
    os.environ['DAM_CODE_VERSION'] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.rstrip()
except Exception as e:
    pass

notebook_name = "Unknown notebook"
try:
    # Not initializing ML trackers yet
    # injected_classes = jvm.io.kensu.third.integration.spark.model.DamModelPublisher.activate().toString()
    ###  Get notebook name ...
    #### see https://github.com/jupyter/notebook/issues/1000#issuecomment-359875246
    import json
    import os.path
    import re
    import requests
    try:  # Python 3
        from urllib.parse import urljoin
    except ImportError:  # Python 2
        from urlparse import urljoin

    def get_notebook_name():
        """
        Return the full path of the jupyter notebook.
        """
        try:  # Python 3
            from notebook.notebookapp import list_running_servers
        except ImportError:  # Python 2
            try:
                import warnings
                from IPython.utils.shimmodule import ShimWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ShimWarning)
                    from IPython.html.notebookapp import list_running_servers
            except ImportError:  # Probably pyspark script is run without IPython/Jupyter
                if not explicit_process_name:
                    print('WARN Unable to automatically extract Jupyter/pyspark notebook name (did you run it without jupyter?)')
                return ['', explicit_process_name or 'Unknown pyspark filename']
        try:
            import ipykernel
            kernel_id = re.search('kernel-(.*).json',
                                    ipykernel.connect.get_connection_file()).group(1)
            servers = list_running_servers()
            for ss in servers:
                response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                        params={'token': ss.get('token', '')})
                for nn in json.loads(response.text):
                    if nn['kernel']['id'] == kernel_id:
                        server = ss
                        notebooks_path = server['notebook_dir']
                        return [notebooks_path, nn['notebook']['path']]
        except Exception as e:
            if not explicit_process_name:
                print('WARN Unable to automatically extract pyspark notebook name')
            return ['', explicit_process_name or 'Unknown pyspark filename']

    notebooks_path, notebook_name = get_notebook_name()
except:
    pass

spark = create_spark_kensu(project, None, "Lab", offline=OFFLINE)
from pyspark import SQLContext
sql = SQLContext(spark)

# DEMO DATASOURCE
if "DB_USER" not in os.environ or "DB_PASSWORD" not in os.environ or "DB_CONNECTION_URL" not in os.environ:
    print("Var env DB_USER or DB_PASSWORD or DB_CONNECTION_URL missing")

notebook_segments = os.path.split(notebook_name)
offline_file_name = notebook_segments[len(notebook_segments)-1]+".jsonl"

dam = DamProvider().initDam(api_url="https://api-demo102.usnek.com", auth_token=token, process_name=notebook_name,
                            user_name=os.environ["USER"], code_location=os.environ['DAM_CODE_REPOSITORY'], 
                            init_context=True, do_report=True, report_to_file=OFFLINE, offline_file_name=offline_file_name,
                            project_names=[project], 
                            pandas_support=True, sklearn_support=True,bigquery_support= True,tensorflow_support=False, 
                            environment="Lab",
                            mapping=True,report_in_mem = False)

def read_pyspark_logs():
    data = []
    with open("dam-offline.log") as f:
        for line in f:
            data.append(json.loads(line))
    return data
def read_pandas_logs():
    data = []
    with open(offline_file_name) as f:
        for line in f:
            data.append(json.loads(line))
    return data
def read_scikit_logs():
    data = []
    with open(offline_file_name) as f:
        for line in f:
            data.append(json.loads(line))
    return data
