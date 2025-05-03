import re
import glob
import multiprocessing
from template import GEN_QUERY_NEW
from Retriever import APIModel
from typing import List
from tqdm import tqdm
from random import sample
from document import TextDocument

def create_query(files_path: List[str], model: APIModel, save_path: str) -> List[List[str]]:
    with open(save_path, "a") as f:
        for file in tqdm(files_path, desc=model.api_name):
            d = TextDocument()
            ent_list = d.input_documents(file, 768)
            chunks = sample(ent_list, 6)
            for chunk in chunks:
                try:
                    query_list = model(context=chunk).split('\n')
                    for q in query_list:
                        q = re.sub(r'^\d\.', '', q)
                        f.write('{{\"query\": \"{}\", \"pos\": [\"{}\"], \"neg\": []}}\n'.format(q, chunk.replace("\n", " ")))
                except Exception as e:
                    print(e)
                    continue

def process(document_path: str):
    file_paths = glob.glob(f'{document_path}/*.md')
    step = APIModel(GEN_QUERY_NEW, "step")
    glm = APIModel(GEN_QUERY_NEW, "glm")

    p_1 = multiprocessing.Process(target=create_query, args=(file_paths[: len(file_paths) // 2], step, "data/ds/query_chunk_step.jsonl"))
    p_2 = multiprocessing.Process(target=create_query, args=(file_paths[len(file_paths) // 2: ], glm, "data/ds/query_chunk_glm.jsonl"))
    p_1.start()
    p_2.start()
    p_1.join()
    p_2.join()
    

if __name__ == "__main__":
    process("data/md")