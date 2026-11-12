#!/usr/bin/env bash
# Zips the project (code + local data/) so it can be uploaded to Google Colab.
set -euo pipefail
cd "$(dirname "$0")/../.."
zip -qr bert-cuda.zip bert-cuda -x 'bert-cuda/bin/*' 'bert-cuda/results/*' 'bert-cuda/.git/*'
echo "created $(pwd)/bert-cuda.zip"
