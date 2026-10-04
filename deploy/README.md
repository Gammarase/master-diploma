# Deploy from Docker Hub images

Runs the Disinformation Detection System (web app + Ollama + SearXNG) from the
pre-built images `gammarase/disinfo-app`, `gammarase/disinfo-ollama` and
`gammarase/disinfo-searxng`. You don't need a repository checkout, only the
files in this folder.

## Requirements

- Docker with Compose v2.24+
- With a GPU: an NVIDIA GPU (about 12 GB of memory for the default `qwen3:14b`)
  and the NVIDIA Container Toolkit
- About 20 GB of free disk space for the images, the LLM and the reranker models

## Run

```sh
mkdir disinfo && cd disinfo
base=https://raw.githubusercontent.com/Gammarase/master-diploma/master/deploy
curl -fsSLO $base/docker-compose.yml
curl -fsSLO $base/docker-compose.no-gpu.yml
curl -fsSL  $base/.env.example -o .env

# Set SEARXNG_SECRET in .env (generate one with: openssl rand -hex 32)
docker compose up -d
```

The raw URLs only work once `deploy/` is pushed to `master` (and only while
the repository is public). Otherwise copy the three files over by hand.

On first start, `ollama` pulls `OLLAMA_MODEL`, which can take a while. `app`
starts once `ollama` and `searxng` are healthy. You can follow progress with
`docker compose logs -f`. The demo page is at `http://localhost:8000/`
(`APP_PORT` in `.env`).

### Without an NVIDIA GPU

```sh
docker compose -f docker-compose.yml -f docker-compose.no-gpu.yml up -d
```

Pick a small model in `.env` (e.g. `OLLAMA_MODEL=qwen3:4b`).

## Update

```sh
docker compose pull
docker compose up -d
```

To pin a specific build instead of `latest`, set `IMAGE_TAG` in `.env`.

## Settings

Every setting is documented in `.env.example`. Models, the evidence cache and
logs live in named volumes, so they survive `docker compose down`. Running
`docker compose down -v` removes them.

## Publishing the images (maintainers)

Publish from the repository root, after `docker login`:

```sh
docker compose --env-file docker/.env build
docker compose --env-file docker/.env push
```
