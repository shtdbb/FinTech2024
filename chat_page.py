#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import atexit
import requests
from copy import deepcopy
from logger import logger
import streamlit as st
from typing import List
from functions import tools, run_python
from VectorDatabase import VectorDatabase
from embedding import EmbeddingAPI, RerankEmbeddingAPI, BM25
from document import TextDocument, PDFDocument, SheetDocument
from Retriever import HyDE, Route, Retriever, SheetRetriever, ChartRetriever
from template import API_LIST, RAG_TEMPLATE_NEW, RAG_TEMPLATE_IE, RAG_TEMPLATE_CHART, DOCUMENT_SELECTED


DOCUMENT_TMP_PATH = "./tmp/"

st.title("文档问答投研 Agent 系统")


def clear_tmp_files(file_list: list):
    for file in file_list:
        if os.path.exists(file): os.remove(file)
        if os.path.exists(".".join(file.split('.')[: -1]) + ".txt"): os.remove(".".join(file.split('.')[: -1]) + ".txt")

# 添加一个重置按钮
if st.sidebar.button('重置系统'):
    clear_tmp_files(st.session_state.file_list)
    st.session_state.clear()

if st.sidebar.button('清空对话'):
    st.session_state.messages = []

# 临时存放文件
if not os.path.exists(DOCUMENT_TMP_PATH):
    os.makedirs(DOCUMENT_TMP_PATH)

if "file_list" not in st.session_state:
    st.session_state.file_list = []


atexit.register(clear_tmp_files, st.session_state.file_list)


def build_knowledge_chat_bot():
    """创建知识问答机器人"""
    def knowledge_chat_bot(query, sents): 
        prompt = RAG_TEMPLATE_NEW.format(
                            context="\n".join([f"[{i+1}] {s.strip()}" for i, s in enumerate(sents)]), 
                            query=query)
        logger.info(f"chat model prompt: {prompt}")
        return requests.post(API_LIST["qwen1half-14b-chat"]["url"], 
            json={"messages": [{"role": "user", "content": prompt}], 
                  "model": "qwen1half-14b-chat", 
                  "stream": False, 
                #   "tools": tools,
                #   "tool_choice": None
                  },
            headers=API_LIST["qwen1half-14b-chat"]["headers"]).json()["choices"][0]["message"]["content"]
    return knowledge_chat_bot


def build_knowledge_ie_bot():
    """创建信息抽取机器人"""
    def knowledge_ie_bot(query, sents): 
        prompt = RAG_TEMPLATE_IE.format(
                            context="\n".join([f"[{i+1}] {s.strip()}" for i, s in enumerate(sents)]), 
                            query=query)
        logger.info(f"ie model prompt: {prompt}")
        return requests.post(API_LIST["qwen1half-14b-chat"]["url"], 
            json={"messages": [{"role": "user", "content": prompt}], 
                  "model": "qwen1half-14b-chat", 
                  "stream": False},
            headers=API_LIST["qwen1half-14b-chat"]["headers"]).json()["choices"][0]["message"]["content"]
    return knowledge_ie_bot


def build_knowledge_chart_bot():
    """创建图表问答机器人"""
    def knowledge_chart_bot(query, sents): 
        prompt = RAG_TEMPLATE_CHART.format(
                            context="\n".join([f"[{i+1}] {s.strip()}" for i, s in enumerate(sents)]), 
                            query=query)
        logger.info(f"chart model prompt: {prompt}")
        return requests.post(API_LIST["qwen1half-14b-chat"]["url"], 
            json={"messages": [{"role": "user", "content": prompt}], 
                  "model": "qwen1half-14b-chat", 
                  "stream": False},
            headers=API_LIST["qwen1half-14b-chat"]["headers"]).json()["choices"][0]["message"]["content"]
    return knowledge_chart_bot


def build_document_selected_bot():
    """创建文档选择机器人"""
    def document_selected_bot(query): 
        prompt = DOCUMENT_SELECTED.format(query=query)
        logger.info(f"document selected model prompt: {prompt}")
        return requests.post(API_LIST["qwen1half-14b-chat"]["url"], 
            json={"messages": [{"role": "user", "content": prompt}], 
                  "model": "qwen1half-14b-chat", 
                  "stream": False},
            headers=API_LIST["qwen1half-14b-chat"]["headers"]).json()["choices"][0]["message"]["content"]
    return document_selected_bot


def markdown_to_plaintext(markdown_text):
    # 移除标题
    plaintext = re.sub(r'#+', '', markdown_text)
    # 移除加粗和斜体
    plaintext = re.sub(r'[*_]{1,3}', '', plaintext)
    # 移除链接
    plaintext = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plaintext)
    # 移除引用
    plaintext = re.sub(r'>+', '', plaintext)
    # 移除无序列表
    plaintext = re.sub(r'- \[ \] ', '', plaintext)
    plaintext = re.sub(r'- [x] ', '', plaintext)
    plaintext = re.sub(r'- ', '', plaintext)
    # 移除有序列表
    plaintext = re.sub(r'\d+\.', '', plaintext)
    # 移除分隔线
    plaintext = re.sub(r'---+', '', plaintext)
    # 移除HTML标签
    plaintext = re.sub(r'<[^>]+>', '', plaintext)
    # 移除多余的空白字符
    plaintext = re.sub(r'(?!<=\d)(?<!\.)\s+', ' ', plaintext).strip()
    return plaintext


def add_cite(response: str, segment_list: List[str]) -> str:
    """
    为回复添加文段参考引用，用于观察召回效果

    Args:
        response (str): 模型的回复
        segment_list (list[str]): 召回的文段列表

    Returns:
        None: 处理后的 md 格式回复
    """
    response = re.sub(r'(\d),(\d)', r'\1\2', response)
    # response = re.sub(r'\^【([0-9]+)†来源】\^', r'[\^{\1}\]', response)   # 单引用格式
    for i in range(len(segment_list)):
        response = response.replace(f"【{i+1}†来源】", f"[^{i+1}]")

    # response = re.sub(r'【([0-9]+)†来源】【([0-9]+)†来源】', r'[\^{\1}][\^{\2}\]', response)   # 多引用格式
    response = response.replace("【1†2†来源】", "[^1][^2]")
    response = response.replace("【1†3†来源】", "[^1][^3]")
    response = response.replace("【2†3†来源】", "[^2][^3]")
    # 文末添加未被模型引用的隐藏引用格式，确保能把未引用的文段也显示出来
    response += f'<span style="visibility:hidden">\
                {"".join([f"[^{i+1}]" for i in range(len(segment_list))])}.</span>'

    cites = []
    # 格式化召回的文段
    for i, s in enumerate(segment_list):
        name = s.split("文档里，目录层级为：", 1)[0]   # 提取文档名
        if name == s:   # 若文档名不符合通用格式
            name = s.split("文档以下为正文内容", 1)[0]
            title = ""   # 这种格式无标题
        else:   # 符合通用格式，提取标题并斜体表示
            title = "*" + s.split("，以下为正文内容：", 1)[0].split("目录层级为：")[-1] + "*"
        content = s.split("以下为正文内容：", 1)[-1].strip("*").strip("#").strip("`").strip("- ").replace("\n", " ").replace('\r', '')   # 提取文段
        content = markdown_to_plaintext(content)
        item = f"[^{i+1}]: 《{name}》：{title}  \n{content}"
        cites.append(item)
    
    response += "  \n" + "  \n".join(cites)
    return response


def display_dialog() -> None:
    """
    渲染对话

    Args:
        segment_list (list[str]):  文段列表

    Returns:
        None: 无返回值
    """
    # 渲染对话内容
    for message in st.session_state.messages:  # 渲染历史记录
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"], unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if "embedding" not in st.session_state:
    st.session_state.embedding = EmbeddingAPI()

if "db_short" not in st.session_state:
    st.session_state.db_short = []

if "db_long" not in st.session_state:
    st.session_state.db_long = []

if "rerank" not in st.session_state:
    st.session_state.rerank = RerankEmbeddingAPI()

if "hyde" not in st.session_state:
    st.session_state.hyde = {}
    st.session_state.hyde["concise"] = HyDE(api_name="concise")
    st.session_state.hyde["glm"] = HyDE(api_name="glm")
    st.session_state.hyde["step"] = HyDE(api_name="step")
    st.session_state.hyde["qwen"] = HyDE(api_name="qwen")
    

if "messages" not in st.session_state:
    st.session_state.messages = []
    

if "document" not in st.session_state:
    st.session_state.document = {}
    st.session_state.document["txt"] = {}
    st.session_state.document["txt"].update({"long": []})
    st.session_state.document["txt"].update({"short": []})
    st.session_state.document["pdf"] = {}
    st.session_state.document["pdf"].update({"long": []})
    st.session_state.document["pdf"].update({"short": []})
    st.session_state.document["sheet"] = []
    
if "knowledge_chat_bot" not in st.session_state:
    knowledge_chat_bot = build_knowledge_chat_bot()
    st.session_state.knowledge_chat_bot = knowledge_chat_bot

if "knowledge_ie_bot" not in st.session_state:
    knowledge_ie_bot = build_knowledge_ie_bot()
    st.session_state.knowledge_ie_bot = knowledge_ie_bot

if "knowledge_chart_bot" not in st.session_state:
    knowledge_chart_bot = build_knowledge_chart_bot()
    st.session_state.knowledge_chart_bot = knowledge_chart_bot

if "document_selected_bot" not in st.session_state:
    document_selected_bot = build_document_selected_bot()
    st.session_state.document_selected_bot = document_selected_bot

if "sheet_chat_bot" not in st.session_state:
    st.session_state.sheet_chat_bot = SheetRetriever()


# 文件上传
st.sidebar.write("\n")
st.sidebar.write("\n")
postfix_type = {"txt": "txt", "md": "txt", "csv": "sheet", "pdf": "pdf", "xlsx": "sheet"}
header = st.sidebar.selectbox('**若上传表格，请先选择表格首行（有列名的行）索引，再上传文件**', [None, 0, 1, 2, 3, 4, 5])
uploaded_files: list = st.sidebar.file_uploader("请上传文件（txt, csv, PDF, Markdown, Excel）", 
                                               type=list(postfix_type.keys()), accept_multiple_files=True)


# 若上传的文件存在
if uploaded_files is not None:
    file_names = [".".join(uploaded_file.name.split(".")[: -1]) + "." + uploaded_file.name.split(".")[-1].lower()
                 for uploaded_file in uploaded_files]

    # 拟保存的文件的完整路径
    file_paths = [os.path.join(DOCUMENT_TMP_PATH, file_name) for file_name in file_names]
    st.session_state.file_list += file_paths
    file_types = [postfix_type.get(file_name.split(".")[-1]) for file_name in file_names]

    # 初始化进度条
    if file_names:
        my_bar = st.progress(0)
    placeholder = st.empty()
    if file_types and len(st.session_state.document[file_types[0]]["long"]) < len(uploaded_files):
        
        # 检查文件是否存在，如果不存在则保存，否则直接提示上传成功
        with st.spinner('正在保存文档 ...'):
            for i, file_path, uploaded_file in zip(range(len(file_paths)), file_paths, uploaded_files):
                if not os.path.exists(file_path):
                    # 保存文件
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
        placeholder.success(f'文档保存完成！', icon="✅")
        my_bar.progress(0.20, "保存文档✅  -→  **文本分段**⏳  -→  段落 Embeddings🕛  -→  存入 Faiss🕛  -→  计算 BM25🕛")
        
        # 分句
        with st.spinner('正在分段 ...'):
            for i, file_path, file_type in zip(range(len(file_paths)), file_paths, file_types):
                if file_type == "sheet":
                    st.session_state.document[file_type].append(SheetDocument.read_sheet(file_path, header))

                else:
                    # 新文档需要导入 document 切分
                    obj_long = PDFDocument() if file_type == "pdf" else TextDocument()
                    sentences_long = obj_long.input_documents(file_path, max_length=768)
                    st.session_state.document[file_type]["long"].append(deepcopy(obj_long))
                    
                    obj_short = PDFDocument() if file_type == "pdf" else TextDocument()
                    sentences_short = obj_short.input_documents(file_path, max_length=384)
                    st.session_state.document[file_type]["short"].append(deepcopy(obj_short))
                    
                    if "sentences_long" not in st.session_state:
                        st.session_state.sentences_long = [sentences_long]
                    else:
                        st.session_state.sentences_long.append(sentences_long)
                    if "sentences_short" not in st.session_state:
                        st.session_state.sentences_short = [sentences_short]
                    else:
                        st.session_state.sentences_short.append(sentences_short)

        placeholder.success('文件分段完成！', icon="✅")
        my_bar.progress(0.40, "保存文档✅  --→  文本分段✅  --→  **段落 Embeddings**⏳  --→  存入 Faiss🕛  --→  计算 BM25🕛") 
            
        # 嵌入
        with st.spinner('正在 Embedding ...'):
            start_len_longs, start_len_shorts = [], []   # [{'id': i, 'len': len(d)}, ...]
            sentences_long, sentences_short = [], []
            i, j = 0, 0
            for d in st.session_state.sentences_long:
                start_len_longs.append({'id': i, 'len': len(d)})
                sentences_long += d
                i += len(d)
            for d in st.session_state.sentences_short:
                start_len_shorts.append({'id': j, 'len': len(d)})
                sentences_short += d
                j += len(d)
                
            long_len = len(sentences_long)
            embeddings = st.session_state.embedding(sentences_long + sentences_short, is_query=False)
            embeddings_long, embeddings_short = embeddings[: long_len], embeddings[long_len: ]
            
            embeddings_longs, embeddings_shorts = [], []
            for d in start_len_longs:
                embeddings_longs.append(embeddings_long[d['id']: d['id'] + d['len']])
            for d in start_len_shorts:
                embeddings_shorts.append(embeddings_short[d['id']: d['id'] + d['len']])

        placeholder.success('Embedding 完成！', icon="✅")
        my_bar.progress(0.60, "保存文档✅  --→  文本分段✅  --→  段落 Embeddings✅  --→  **存入 Faiss**⏳  --→  计算 BM25🕛")
        
        # 存入 Faiss
        with st.spinner('正在存入 Faiss ...'):
            for i, embeddings_long, embeddings_short in zip(range(len(embeddings_longs)), embeddings_longs, embeddings_shorts):
                obj_long = VectorDatabase(n_hiddens=1024, db_name=f"long_{file_names[i]}")
                obj_long.create_index(embeddings_long)
                st.session_state.db_long.append(obj_long)
                
                obj_short = VectorDatabase(n_hiddens=1024, db_name=f"short_{file_names[i]}")
                obj_short.create_index(embeddings_short)
                st.session_state.db_short.append(obj_short)
        placeholder.success('存入 Faiss 完成！', icon="✅")
        my_bar.progress(0.80, "保存文档✅  --→  文本分段✅  --→  段落 Embeddings✅  --→  存入 Faiss✅  --→  **计算 BM25**⏳")  
        
        # 计算 BM25
        with st.spinner('正在计算 BM25 ...'):
            for sentences_long, sentences_short in zip(st.session_state.sentences_long, st.session_state.sentences_short):
                if "bm25_long" not in st.session_state:
                    st.session_state.bm25_long = [BM25(sentences_long)]
                else:
                    st.session_state.bm25_long.append(BM25(sentences_long))
                if "bm25_short" not in st.session_state:
                    st.session_state.bm25_short = [BM25(sentences_short)]
                else:
                    st.session_state.bm25_short.append(BM25(sentences_short))
        placeholder.success('BM25 计算完成！', icon="✅")
        my_bar.progress(1.00, "保存文档✅  --→  文本分段✅  --→  段落 Embeddings✅  --→  存入 Faiss✅  --→  计算 BM25✅")
            
        placeholder.success(f"{len(st.session_state.document[file_types[0]]['long'])} 个文档处理完成！", icon="✅")

# 显示历史对话
display_dialog()
# 若用户提问
if user_query := st.chat_input("您好，请问有什么可以帮到您？"):
    # 文档选择
    try:
        idx = int(st.session_state.document_selected_bot(user_query))
    except:
        idx = 0
    logger.info(f"document selected index: {['未提及时间', '提到一个时间', '提到两个时间', '提到四个时间'][idx]}")    
    
    if idx in [0, 1]:   # 选择单文档
        file_id = [st.session_state.rerank(user_query if idx == 1 else user_query + "，请根据年度报告", file_names)[0]] if len(file_names) > 1 else 0
    elif idx == 1:   # 两个文档
        file_id = st.session_state.rerank(user_query, file_names)[: 2] if len(file_names) > 1 else [0]
    else:   # 四个文档
        file_id = st.session_state.rerank(user_query + "第一季度、第二季度、第三季度、第四季度", file_names)[: 4] if len(file_names) > 1 else [0]
    
    # 意图识别 0-金融问答计算, 1-json信息抽取, 2-统计图表绘制
    intention = st.session_state.rerank(user_query, ["金融问答计算", "json信息抽取", "统计图表绘制"])[0]
    logger.info(f"query intention: {['金融问答计算', 'json信息抽取', '统计图表绘制'][intention]}")
    
    
    # 显示并存储提问
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    if len(file_id) == 1:
        st.info(f"匹配单文档：{file_names[file_id[0]]}", icon="📄")
    else:
        st.info(f"匹配多文档：{'、'.join([file_names[i] for i in file_id])}", icon="📄")
    
    if file_types[file_id[0]] == "sheet":   # 表格问答
        response = st.session_state.sheet_chat_bot(user_query, st.session_state.document["sheet"][file_id[0]])
        logger.info(f"chat model response: {response}")
    else:
        # 配置多路召回
        routes = []
        for i in file_id:
            routes += [
                Route(8, st.session_state.embedding, st.session_state.db_long[i], None, None),
                Route(8, None, None, st.session_state.bm25_long[i], None),
                Route(8, st.session_state.embedding, st.session_state.db_short[i], None, None),
                Route(8, None, None, st.session_state.bm25_short[i], None),
                # Route(3, st.session_state.embedding, st.session_state.db_long, None, st.session_state.hyde["concise"]),
                # Route(3, None, None, st.session_state.bm25_long, st.session_state.hyde["glm"]),
                # Route(5, st.session_state.embedding, st.session_state.db_short, None, st.session_state.hyde["step"]),
                # Route(5, None, None, st.session_state.bm25_short, st.session_state.hyde["qwen"])
        ]
        retriever = Retriever(routes, st.session_state.rerank)

        sentences = []
        for i in file_id:
            sentences += [st.session_state.sentences_long[i], st.session_state.sentences_long[i],
                       st.session_state.sentences_short[i], st.session_state.sentences_short[i],
                        # st.session_state.sentences_long, st.session_state.sentences_long,
                        # st.session_state.sentences_short, st.session_state.sentences_short
                        ]

        segment_list = retriever(user_query, 
                                 sentences,
                                 topk=4,
                                 lost_in_the_middle=True)   # 避免 lost in the middle
        
        logger.info("segment_list: " + "\n\n".join(segment_list))
        
        # 返回模型回复
        if intention == 0:   # 知识问答
            response = st.session_state.knowledge_chat_bot(user_query, segment_list)
        elif intention == 1:   # 信息抽取
            response = st.session_state.knowledge_ie_bot(user_query, segment_list)
            output = re.findall(r'```json(.*?)```', response, re.DOTALL)
            response = output[0] if output else response
            # output = re.findall(r'\{.*?\}', response, re.DOTALL)
            # response = output[0] if output else response
        else:   # 图表
            response = st.session_state.knowledge_chart_bot(user_query, segment_list)
            logger.info(f"chart description: {response}")
            cr = ChartRetriever()
            response = cr(user_query, response)
            if "![img](./tmp/figure.png)" in response:
                st.image("tmp/figure.png", use_column_width=True)
                response = response.replace("![img](./tmp/figure.png)", "")   # streamlit不支持md格式的图片展示
        
        logger.info(f"{'chat' if intention == 0 else ('ie' if intention == 1 else 'chart')} model response: {response}")
        
        # 处理输出格式
        response = add_cite(response, segment_list)

    # 存储输出内容
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 显示输出
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
