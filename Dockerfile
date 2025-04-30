# On commence par définir l'image de base avec une version légère de Python 3.10
ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE as runtime-environment

# Mise à jour des paquets et installation de gcc et build-essential pour pouvoir compiler certains paquets Python natifs
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# Mise à jour de pip à une version récente pour une meilleure compatibilité
RUN python -m pip install -U "pip>=21.2"

# Installation de l'outil 'uv', qui permet une installation plus rapide et moderne des paquets Python
RUN pip install uv

# Copie du fichier des dépendances dans le conteneur
COPY requirements.txt /tmp/requirements.txt

# Installation des dépendances du projet à partir du requirements.txt, puis suppression du fichier temporaire
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Création d'un utilisateur dédié (kedro_docker) pour exécuter le projet, ce qui est une bonne pratique de sécurité
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

# Définition du répertoire de travail pour l'utilisateur kedro_docker
WORKDIR /home/kedro_docker
USER kedro_docker

# Deuxième phase du Dockerfile : on repart de l’environnement runtime pour inclure le code de l'application
FROM runtime-environment

# Recopie tous les fichiers du projet (en respectant les exclusions définies dans .dockerignore) avec les bons droits utilisateur
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .
RUN mkdir -p /home/kedro_docker/data

# On expose le port 8888, utile si on lance un notebook ou une API
EXPOSE 8888

# Commande par défaut : on exécute le pipeline Kedro
CMD ["kedro", "run"]