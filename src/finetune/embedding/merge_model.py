import os
from config import config
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from LM_Cocktail import mix_models

raw_model = config.BGE_EMBEDDING_PATH
finetuned_model = "./resources/model/checkpoint_xxxx"   # 微调后模型路径
weights = [0.2, 0.8]   # 原模型和微调模型的权重

model = mix_models(
    model_names_or_paths=[raw_model, finetuned_model],
    model_type='encoder',
    # model_type='reranker',   # 若融合 rerenker 模型
    weights=weights,
    output_path=f'./model/mixed_model_{finetuned_model.split("/")[-1]}_' + '_'.join([str(w) for w in weights]))

