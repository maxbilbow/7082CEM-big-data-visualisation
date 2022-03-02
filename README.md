# 7082CEM-big-data-visualisation

## Getting Started

### Install Docker
Docker is available from [docker.io](https://docker.io).

Using Homebrew (MacOS): `brew install --cask docker`

### Build and start the containers

```shell
docker-compose up -d
```

### Get Token for Jupyter Notebook

1. Connect to the Jupyter container:
```shell
docker exec -it bdv_notebook /bin/bash
```

2. Find the token:
```shell
jupyter server list
```
This should pring something like (the token will differ): 
```
Currently running servers:
http://fea85ddab0f9:8888/?token=933a00cfab499febdd295986a91f63ea209a7b86cf89e131 :: /home/jovyan
```

3. Open the Notebooks
Navigate to `http://localhost:8888/?token={whatever_you_token_is}`

## Running Jupyter Notebooks as Python Scripts
To run our Jupyter notebooks on our Spark container, we must first convert them to .py Python files.

1. Connect to the Jupyter container:
```shell
docker exec -it bdv_notebook /bin/bash
```

2. Convert all notebook files:
```shell
jupyter nbconvert --to script work/*.ipynb
```

## Run scripts on Spark container
1. Connect to the container:
```shell
docker exec -it bdv_spark /bin/bash
```

2. Run the script:
```shell
spark-submit LogisticRegression.py
```