#!/usr/bin/env bash

set -e


export GCP_PROJECT=dae-intern-2024-04
# Verify that all required variables are set
if [[ -z "${GCP_PROJECT}" ]]; then
    echo "Error: GCP_PROJECT not set"
    exit 1
fi

# Fetch secrets from Secret Manager in CGP
export MLFLOW_TRACKING_USERNAME="$(python3 get_secret.py --project="${GCP_PROJECT}" --secret=dae-intern-mlflow-username)"
export MLFLOW_TRACKING_PASSWORD="$(python3 get_secret.py --project="${GCP_PROJECT}" --secret=dae-intern-mlflow-password)"
export ARTIFACT_URL="$(python3 get_secret.py --project="${GCP_PROJECT}" --secret=dae-intern-mlflow-artifact-url)"
if [[ -z "${DATABASE_URL}" ]]; then # Allow overriding for local deployment
    export DATABASE_URL="$(python3 get_secret.py --project="${GCP_PROJECT}" --secret=dae-intern-mlflow-database-url)"
fi


if [[ -z "${MLFLOW_TRACKING_USERNAME}" ]]; then
    MLFLOW_TRACKING_USERNAME=dea-intern-202404
fi

if [[ -z "${MLFLOW_TRACKING_PASSWORD}" ]]; then
    MLFLOW_TRACKING_PASSWORD=Arizona123!
fi

if [[ -z "${ARTIFACT_URL}" ]]; then
    ARTIFACT_URL=gs://dae-intern-mlflow-artifact-url
fi

if [[ -z "${DATABASE_URL}" ]]; then
    DATABASE_URL=postgresql+psycopg2://dae-intern-2024-04:TokyoTech1998!@/dae-intern?host=/cloudsql/dae-intern-2024-04:us-central1:macromill-intern-mlflow-trail
fi

if [[ -z "${PORT}" ]]; then
    export PORT=8080
fi

if [[ -z "${HOST}" ]]; then
    export HOST=0.0.0.0
fi

export WSGI_AUTH_CREDENTIALS="${MLFLOW_TRACKING_USERNAME}:${MLFLOW_TRACKING_PASSWORD}"
export _MLFLOW_SERVER_ARTIFACT_ROOT="${ARTIFACT_URL}"
export _MLFLOW_SERVER_FILE_STORE="${DATABASE_URL}"

# Start MLflow and ngingx using supervisor
exec gunicorn -b "${HOST}:${PORT}" -w 4 --log-level debug --access-logfile=- --error-logfile=- --log-level=debug mlflow_auth:app