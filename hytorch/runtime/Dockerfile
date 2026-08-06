FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hytorch-pi
COPY package.json hytorch-pi.mjs ./
RUN npm install --omit=dev --no-audit --no-fund

ENTRYPOINT ["node", "/opt/hytorch-pi/hytorch-pi.mjs"]
