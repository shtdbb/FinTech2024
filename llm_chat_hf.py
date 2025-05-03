# -*- coding: utf-8 -*-
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# from config.tools import tools
from typing import Union, List, Tuple
from transformers import (AutoTokenizer, AutoModel, 
                          AutoModelForCausalLM, 
                          AutoModelForSequenceClassification)


class Model:
    """
    HuggingFace 模型基类
    """
    model_list = {
        "THUDM/chatglm-6b": "chatglm",
        "THUDM/chatglm2-6b": "chatglm",
        "THUDM/chatglm3-6b": "chatglm",
        "Qwen/Qwen-7B-Chat": "qwen",
        "Qwen/Qwen-14B-Chat": "qwen",
        "BAAI/bge-large-zh": "bert",
        "BAAI/bge-large-zh-v1.5": "bert",
        "BAAI/bge-reranker-large": "xlm-roberta",
        "BAAI/bge-reranker-base": "xlm-roberta"
    }
    
    @classmethod
    def get_model_type(cls, model_name_or_path: str) -> None:
        if model_name_or_path in cls.model_list:
            cls.model_type = cls.model_list[model_name_or_path]
            return
        
        with open(f'{model_name_or_path}/config.json', 'r') as f:
            data = json.load(f)
        cls.model_type = data['model_type']
        return
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path: str, 
                        device: str = "cuda", 
                        int_4: bool = False
                        ) -> object:
        """加载模型

        Args:
            model_name_or_path (str): 模型的 HuggingFace 仓库名或本地路径\n
            device (str, optional): 加载设备, "cuda": 单 GPU, "auto": 多 GPU, "cpu": CPU. Defaults to "cuda".\n
            int_4 (bool, optional): 是否开启 int4 量化. Defaults to False.

        Returns:
            object: 模型对象
        """
        assert device in ["auto", "cuda", "cpu"], \
            f"`device` must be 'auto' or 'cuda', not support cpu currently. Your input is: {device}"
        
        cls.model_name_or_path = model_name_or_path
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name_or_path, trust_remote_code=True)

        cls.get_model_type(model_name_or_path)
        if cls.model_type in ["chatglm", "bert"]:
            if device == "auto":
                if int_4:
                    cls.model = AutoModel.from_pretrained(cls.model_name_or_path, 
                                                trust_remote_code=True, device_map="auto").quantize(4)
                else:
                    cls.model = AutoModel.from_pretrained(cls.model_name_or_path, 
                                                trust_remote_code=True, device_map="auto")
            elif device == "cuda":
                if int_4:
                    assert cls.model_type != "bert", "目前嵌入模型不支持量化"
                    cls.model = AutoModel.from_pretrained(cls.model_name_or_path, 
                                                trust_remote_code=True).half().quantize(4).cuda()
                else:
                    cls.model = AutoModel.from_pretrained(cls.model_name_or_path, 
                                                trust_remote_code=True).half().cuda()
            else:
                cls.model = AutoModel.from_pretrained(cls.model_name_or_path, 
                                                trust_remote_code=True).float()

            cls.model = cls.model.eval()
            
        elif cls.model_type == "qwen":
            assert int_4 == False, "Qwen 模型只支持原生 int4 模型权重加载, 不支持从半精度转换, \
                                            请下载并加载 Qwen 原生 int4 模型权重."
            
            if device == "auto":
                cls.model = AutoModelForCausalLM.from_pretrained(
                    cls.model_name_or_path, trust_remote_code=True, device_map="auto", fp16=True).eval()
            elif device == "cuda":
                cls.model = AutoModelForCausalLM.from_pretrained(
                    cls.model_name_or_path, trust_remote_code=True, fp16=True).cuda().eval()
            else:
                cls.model = AutoModelForCausalLM.from_pretrained(
                    cls.model_name_or_path, trust_remote_code=True).eval()
        
        elif cls.model_type == "xlm-roberta":
            assert int_4 == False, "目前嵌入模型不支持量化"
            if device == "auto":
                cls.model = AutoModelForSequenceClassification.from_pretrained(
                    cls.model_name_or_path, trust_remote_code=True, device_map="auto").eval()
            elif device == "cuda":
                cls.model = AutoModelForSequenceClassification.from_pretrained(
                    cls.model_name_or_path, trust_remote_code=True).cuda().eval()
            else:
                cls.model = AutoModelForSequenceClassification.from_pretrained(
                    cls.model_name_or_path, trust_remote_code=True).eval()
        
        else:   # TODO baichuan2
            pass
        return cls()


class LLM(Model):
    """
    大语言模型类
    """
    system_prompt = {
        "chatglm": 
            '{"role": "system", "content": "Answer the following questions as best as you can. \
            You have access to the following tools:", "tools": {tools}}',
        "qwen_tool_desc": 
            "{name_for_model}: Call this tool to interact with the {name_for_human} API. \
            What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} \
            Format the arguments as a JSON object.",
        "qwen_react_prompt":
            "Answer the following questions as best you can. You have access to the following tools:\
            \n\n{tool_descs}\n\nUse the following format:\n\nQuestion: the input question you must answer\
            \nThought: you should always think about what to do\
            \nAction: the action to take, should be one of [{tool_names}]\
            \nAction Input: the input to the action\nObservation: the result of the action\
            \n... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\
            \nThought: I now know the final answer\
            \nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {query}"
        }

    def get_qwen_prompt(self, query: str, tools: list) -> str:
        tool_descs, tool_names = [], []

        for info in tools:
            tool_descs.append(
                self.system_prompt["qwen_tool_desc"].format(
                    name_for_model=info['name_for_model'],
                    name_for_human=info['name_for_human'],
                    description_for_model=info['description_for_model'],
                    parameters=json.dumps(
                        info['parameters'], ensure_ascii=False),
                )
            )
            tool_names.append(info['name_for_model'])
            
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        prompt = self.system_prompt["qwen_react_pormpt"].format(
            tool_descs=tool_descs, tool_names=tool_names, query=query)
        return prompt
    
    def chat_completion(self, 
                        query: str, 
                        history: List[str,] = [], 
                        tools: list = None,
                        observation: bool = False
                        ) -> Tuple[str, List[str,]]:
        """对话

        Args:
            query (str): 用户提问\n
            history (List[str,]): 历史对话, 默认为空\n
            tools (list, optional): 可用工具列表, 默认为 None\n
            observation (bool, optional): `query` 是否为 tool 返回值, 默认为 False

        Returns:
            Tuple[str, List[str,]]: response, history
        """
        assert type(query) == str, f"`query` must be str, not {type(query)}"
        
        if self.model_type == "chatglm":   # ChatGLM
            if tools:
                history = [eval(self.system_prompt["chatglm"].format(tools=tools))] + history
                
            response, history = self.model.chat(self.tokenizer, 
                                                query, 
                                                history=history,
                                                role="observation" if observation else "user")
            
        elif self.model_type == "qwen":   # Qwen
            if observation:
                react_stop_words = [
                    # self.tokenizer.encode('Observation'),  # [37763, 367]
                    self.tokenizer.encode('Observation:'),  # [37763, 367, 25]
                    self.tokenizer.encode('Observation:\n'),  # [37763, 367, 510]
                ]
                self.prompt += str(query)
                response, history = self.model.chat(self.tokenizer,
                                                    self.prompt, 
                                                    history, 
                                                    stop_words_ids=react_stop_words)
            else:
                if tools:
                    self.prompt = self.get_qwen_prompt(query, tools)
                    query = self.prompt
                    
                response, history = self.model.chat(self.tokenizer,
                                                    query, 
                                                    history=history)
        else:   # TODO baichuan2
            pass
        
        return response, history


class Embedding(Model):
    """
    双塔嵌入模型类
    """
    def get_text_embedding(self, text_list: List[str,]) -> torch.float16:
        """获取文本嵌入

        Args:
            text_list (List[str,]): 文本列表

        Returns:
            torch.float16: 文本嵌入, shape: (n_texts, embedding_dim)
        """
        encoded_input = self.tokenizer(text_list, 
                                       padding=True, 
                                       truncation=True, 
                                       return_tensors='pt'
                                       ).to(self.model.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings


class Reranker(Model):
    """
    单塔重排序嵌入模型类
    """
    def get_pair_score(self, pairs: List[List[str,],]) -> torch.float32:
        """获取问答对相似度分数

        Args:
            pairs (List[List[str, str],]): 问答对, 如:   
                `[['你叫什么名字？', '我叫小明。'], ['你叫什么名字？', '三体是由刘慈欣所著的长篇科幻小说。']]`

        Returns:
            torch.float32: 问答对的相似度分数, shape: (n_pairs,)
        """
        with torch.no_grad():
            inputs = self.tokenizer(pairs, 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors='pt', 
                                    max_length=512
                                    ).to(self.model.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        return scores


if __name__ == "__main__":
    # 大模型问答
    model = LLM.from_pretrained("models/qwen-14b")
    outs = model.chat_completion("你好")
    print(outs)
    # ('你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。', 
    #   [{'role': 'user', 'content': '你好'}, 
    #    {'role': 'assistant', 'metadata': '', 'content': '你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。'}
    #   ]
    # )
    
    # 文本嵌入
    embedding = Embedding.from_pretrained("models/bge-large-zh")
    outs = embedding.get_text_embedding(["你好，我是小明", "今年是2024年"])
    print(outs, outs.shape)
    # tensor([[ 0.0271, -0.0481, -0.0304,  ..., -0.0145, -0.0151,  0.0016],
    #    [ 0.0103,  0.0007, -0.0489,  ...,  0.0454,  0.0270, -0.0220]], device='cuda:0', dtype=torch.float16) 
    # torch.Size([2, 1024])
    
    # 问答对重排序相似度
    reranker = Reranker.from_pretrained("models/bge-reranker-large")
    outs = reranker.get_pair_score([["你叫什么名字？", "我叫小明。"], ["你叫什么名字？", "三体是由刘慈欣所著的长篇科幻小说。"]])
    print(outs)
    # tensor([-1.8043, -9.4000], device='cuda:0')