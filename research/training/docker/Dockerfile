# TinyTTS training image -- see docs/training.md ("Docker").
#
# Build context is research/ (so the image can include
# export_weights_and_vectors.py next to training/):
#
#   docker build -f research/training/docker/Dockerfile -t tinytts-train:cpu research
#   docker build -f research/training/docker/Dockerfile --build-arg TORCH_VARIANT=cu126 -t tinytts-train:cu126 research
#
# Layout inside the image mirrors the repo, so every script's relative paths
# work unchanged:
#   /research/tiny-tts/     upstream model code (pinned commit, tiny_tts/ only)
#   /research/training/     this pipeline; data/ and runs/ are meant to be mounted
#   /research/export_weights_and_vectors.py
FROM python:3.12-slim

# cpu (default, also arm64) or a CUDA wheel variant from
# https://download.pytorch.org/whl/ -- cu126 runs on the widest range of
# NVIDIA drivers; cu130 for the newest GPUs.
ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHAUDIO_VERSION=2.11.0
ARG UPSTREAM_REPO=https://github.com/tronghieuit/tiny-tts.git
ARG UPSTREAM_COMMIT=77c278f7a4659181bd7bcf5c5e07c768ae765147
# 1 = bake the released backtracking/tiny-tts G.pth into the image so
# `--init pretrained` works offline; 0 = download it on first use instead.
ARG BAKE_CHECKPOINT=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    NLTK_DATA=/opt/nltk_data \
    HOME=/tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
COPY training/docker/requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock

# Upstream model code at a pinned commit. Sparse + blobless: the upstream
# repo also carries large checkpoints/ONNX files that aren't needed here.
RUN git init -q /research/tiny-tts \
    && cd /research/tiny-tts \
    && git remote add origin ${UPSTREAM_REPO} \
    && git sparse-checkout set tiny_tts \
    && git fetch -q --depth 1 --filter=blob:none origin ${UPSTREAM_COMMIT} \
    && git checkout -q FETCH_HEAD \
    && rm -rf .git

COPY training/ /research/training/
COPY export_weights_and_vectors.py /research/

# Pre-fetch everything the pipeline would otherwise download on first use:
# NLTK data (g2p_en), the bert-base-uncased tokenizer (upstream's English
# word splitting), and optionally the released checkpoint. Then smoke-test
# the English frontend so a broken image fails at build time, not mid-run.
RUN python -m nltk.downloader -q -d ${NLTK_DATA} averaged_perceptron_tagger_eng averaged_perceptron_tagger cmudict \
    && cd /research/training \
    && python -c "from frontends import get_frontend; print(get_frontend('EN').text_to_ids('Hello world.')[0])" \
    && if [ "${BAKE_CHECKPOINT}" = "1" ]; then \
         python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('backtracking/tiny-tts', 'G.pth'))"; \
       fi \
    && chmod -R a+rwX ${HF_HOME} ${NLTK_DATA} \
    && chmod +x /research/training/docker/entrypoint.sh

WORKDIR /research/training
ENTRYPOINT ["/research/training/docker/entrypoint.sh"]
CMD ["help"]
