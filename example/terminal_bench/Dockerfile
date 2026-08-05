FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        build-essential \
        ca-certificates \
        curl \
        file \
        git \
        jq \
        openssl \
        python3 \
        python3-pip \
        ripgrep \
        sqlite3 \
        unzip \
        vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hytorch-pi
COPY hytorch/runtime/package.json hytorch/runtime/hytorch-pi.mjs ./
RUN npm install --omit=dev --no-audit --no-fund

ENTRYPOINT ["node", "/opt/hytorch-pi/hytorch-pi.mjs"]
