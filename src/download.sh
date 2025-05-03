#!/bin/bash

pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

mkdir models

mkdir models/bge-reranker-large
huggingface-cli download --resume-download --local-dir-use-symlinks False BAAI/bge-reranker-large --local-dir models/bge-reranker-large

mkdir models/bge-large-zh
huggingface-cli download --resume-download --local-dir-use-symlinks False BAAI/bge-large-zh --local-dir models/bge-large-zh

mkdir models/chatglm3-6b
huggingface-cli download --resume-download --local-dir-use-symlinks False THUDM/chatglm3-6b --local-dir models/chatglm3-6b

# mkdir models/ltp-tiny
# huggingface-cli download --resume-download --local-dir-use-symlinks False LTP/tiny --local-dir models/ltp-tiny
