import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from logger import logger
from typing import List, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from FlagEmbedding import FlagModel, FlagReranker


app = FastAPI()

#embedding_path = "../models/bge-large-zh-v1.5"
embedding_path = "../models/checkpoint_2024_04_27_17_47_55"
logger.info(f"Loading embedding model: {embedding_path} ...")
embed = FlagModel(embedding_path, 
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True)
#reranker_path = "../models/bge-reranker-v2-minicpm-layerwise"
reranker_path = "../models/bge-reranker-large"
logger.info(f"Loading reranker model: {reranker_path} ...")
rerank = FlagReranker(reranker_path, use_fp16=True)


class Embedding(BaseModel):
    sentences: List[str]
    is_query: bool = False

class Reranker(BaseModel):
    query: str
    sentences: List[str]
    topk: int = 3
    normalize: bool = False
    cutoff_layers: Any = None
    return_scores: bool = False


@app.post("/embedding")
async def get_embedding(embedding: Embedding):
    sentences, is_query =  embedding.sentences, embedding.is_query
    if is_query:
        return {"embeddings": 
            embed.encode_queries(sentences if type(sentences) is list else [sentences], 
                                 512, 128).tolist()}
    else:
        return {"embeddings": 
            embed.encode(sentences if type(sentences) is list else [sentences], 
                         512, 512).tolist()}


@app.post("/reranker")
async def get_order(reranker: Reranker):
    query = reranker.query
    sentences = reranker.sentences
    topk = reranker.topk
    normalize = reranker.normalize
    cutoff_layers = reranker.cutoff_layers
    return_scores = reranker.return_scores
    
    assert cutoff_layers == None or (cutoff_layers != None and "layerwise" in reranker_path), "cutoff_layers only works for layerwise reranker."
    pairs = [[query, s] for s in sentences]
    
    if "layerwise" in reranker_path:
        scores = rerank.compute_score(pairs, cutoff_layers=cutoff_layers if not cutoff_layers else [28], normalize=normalize)
    else:
        scores = rerank.compute_score(pairs, normalize=normalize)
        
    sent_order = sorted([(i, s) for i, s in enumerate(scores)], key=lambda x: x[1], reverse=True)
    return {"indexes": [s[0] for s in sent_order[: topk]]} if not return_scores else {"scores": scores}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
