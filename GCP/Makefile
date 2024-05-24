IMAGE_NAME=dae-intern-mlflow
GCP_PROJECT=dae-intern-2024-04
AR_REGISTRY_NAME=dae-intern-2024
AR_REGION=us-central1
VERSION=latest

build:
	docker build -t "${IMAGE_NAME}" .
	
build-m1:
	docker build --platform linux/amd64 -t "${IMAGE_NAME}" .

docker-auth:
	gcloud auth configure-docker

tag:
	docker tag "${IMAGE_NAME}" "${AR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REGISTRY_NAME}/${IMAGE_NAME}:${VERSION}"

push:
	docker push "${AR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REGISTRY_NAME}/${IMAGE_NAME}:${VERSION}"
