# 基于LangChain的大模型RAG智能问答系统 - 学习笔记

> 终态流水线：Greeting → TF-IDF FAQ → 多路RAG(ES∥向量→RRF→重排阈值+会话历史) → 联网 → Fallback

---

## 目录

- [一、课程总览](#一课程总览)
- [二、环境配置](#二环境配置)
- [三、LangChain](#三langchain)
- [四、基础知识](#四基础知识)
- [五、项目初始化](#五项目初始化)
- [六、FAQ问答](#六faq问答)
- [七、RAG知识库](#七rag知识库)
- [八、联网与兜底](#八联网与兜底)
- [九、项目工程化](#九项目工程化)
- [十、评估与进阶](#十评估与进阶)
- [十一、项目拓展](#十一项目拓展)
- [十二、前后端拓展](#十二前后端拓展)
- [附录](#附录)

---

## 一、课程总览

### 1.1 课程概述

- **课程标题**：基于LangChain的大模型RAG智能问答系统
- **核心知识点**：LangChain、RAG
- **讲师**：陈华（6年前后端研发+7年算法研发+8年讲师）
- **研发经历**：
  - 2013-2019：前后端项目研发（数据量过亿级、日请求千万级）
  - 2019-2023：传统算法研发（证照识别、信息抽取、情感分析、风控算法、推荐算法）
  - 2023-2026：大模型算法研发（智能客服、大模型网关、多模态内容生成）

### 1.2 项目简介

**项目定位**：教育场景智能助教系统，覆盖课程咨询FAQ + 学科知识库 + FastAPI对话页

**终态流水线**：
```
Greeting → TF-IDF FAQ → 多路RAG(ES∥向量→RRF→重排阈值+会话历史) → 联网 → Fallback
```

**RAG原理**（Retrieve-Augment-Generate）：
- **Retrieve（检索）**：从知识库中检索与用户问题语义最相关的文档块
- **Augment（增强）**：将检索到的上下文拼入Prompt，增强大模型回答的事实性
- **Generate（生成）**：大模型基于上下文生成答案

**为什么需要RAG**：纯大模型对私有文档/课程大纲等「库内事实」容易答错或胡编，RAG相当于开卷考试

**三库分工**：

| 数据库 | 职责 | 典型场景 |
|--------|------|----------|
| MySQL | 结构化精确查询 | FAQ问答对、会话消息、业务配置 |
| Milvus | 语义向量TopK检索 | RAG知识库文档块检索 |
| Elasticsearch | 全文关键词检索 | BM25关键词召回 |

**技术栈**：

| 层级 | 技术 | 作用 |
|------|------|------|
| Web | FastAPI + Jinja2 + Uvicorn | 页面渲染与问答API |
| 前端 | Tailwind CSS + Alpine.js + marked.js | 多会话UI、阶段标签、Markdown渲染 |
| 编排 | QAPipeline | Greeting → TF-IDF → RAG → Web → Fallback |
| FAQ检索 | jieba + sklearn TF-IDF | 关键词匹配FAQ |
| 向量库 | Milvus | RAG向量检索 |
| 模型 | DashScope（LLM/Embedding） | 生成与向量化 |
| 联网 | SerpAPI | 知识库未命中时联网检索 |
| CLI | argparse + commands/ | 建库、入库、试问等运维命令 |

**项目结构**：
```
edu_rag_project/
├── app.py                 # FastAPI 启动入口
├── routes.py              # 首页渲染 + /api/ask
├── cli.py                 # CLI 命令入口
├── config.py / .env       # 配置与环境变量
├── logger.py              # 日志模块
├── prompts.py             # RAG / WEB / FALLBACK Prompt
├── pipeline.py            # 问答编排入口
├── requirements.txt
├── db/                    # SQLAlchemy Base / session / models
├── alembic/               # 迁移脚本
├── clients/               # milvus / llm / embedding / rerank / elasticsearch
├── services/              # greeting / faq / tfidf / rag / hybrid / web_search / fallback
├── commands/              # ingest_faq / init_milvus / ingest_rag / test_*（业务 CLI）
├── utils/                 # chunking / preprocess
├── templates/             # index.html / admin*.html
└── storage/
    ├── data/              # FAQ / 学科问答 / 大纲 / 停用词
    └── logs/
```

---

## 二、环境配置

### 2.1 Python环境

- 推荐使用 Conda 创建虚拟环境
- Python版本：3.10.x
- IDE推荐：VSCode、PyCharm，后期可接触Cursor、Trae等AI辅助编辑器

```bash
conda create -n edu_rag_project python==3.10.18
conda activate edu_rag_project
pip install -r requirements.txt
```

### 2.2 Docker环境配置

**Docker Compose** 定义以下服务：

| 服务 | 镜像 | 端口 | 用途 |
|------|------|------|------|
| etcd | quay.io/coreos/etcd:v3.5.5 | - | Milvus元数据存储 |
| minio | minio/minio:RELEASE.2023-03-20T20-16-18Z | - | Milvus对象存储 |
| milvus-standalone | milvusdb/milvus:v2.5.0-beta | 19530, 9091 | 向量数据库 |
| mysql | mysql:8.0.33 | 3306 | 关系型数据库 |
| attu | zilliz/attu:v2.6.5 | 3010 | Milvus可视化管理 |

**Docker Compose配置**：
```yaml
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_ENABLE_V2=true
    volumes:
      - ./volumes/etcd:/etcd/data
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd/data

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./volumes/minio:/minio/data
    command: minio server /minio/data

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.5.0-beta
    command: ["milvus", "run", "standalone"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

  mysql:
    container_name: mysql
    image: mysql:8.0.33
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: root123456
      MYSQL_DATABASE: edu_rag_project
      MYSQL_USER: edu_rag
      MYSQL_PASSWORD: app123456
      TZ: Asia/Shanghai
    ports:
      - "3306:3306"
    volumes:
      - ./volumes/mysql/data:/var/lib/mysql
      - ./volumes/mysql/conf:/etc/mysql/conf.d
      - ./volumes/mysql/init:/docker-entrypoint-initdb.d
    command:
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci
      - --default-time-zone=+08:00
      - --max_connections=1000
      - --innodb_buffer_pool_size=256M
      - --innodb_log_file_size=64M

  attu:
    container_name: attu
    image: zilliz/attu:v2.6.5
    restart: always
    ports:
      - "3010:3000"
    depends_on:
      - standalone
```

### 2.3 Ollama模型部署

- Ollama：本地大模型部署工具
- 支持拉取和运行开源大模型（如qwen2.5:7b）
- 提供OpenAI兼容API（默认端口11434）

```bash
# 拉取模型
ollama pull qwen2.5:7b

# 运行模型
ollama run qwen2.5:7b
```

Ollama兼容OpenAI SDK调用：
```python
# config.json 配置
{
    "provider": "ollama",
    "api_key": "",
    "base_url": "http://localhost:11434/v1",
    "model": "qwen2.5:7b",
    "timeout": 120.0
}
```

### 2.4 千问API开通

- 阿里云DashScope提供千问系列模型API
- 兼容OpenAI SDK接口
- 环境变量配置：

```env
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_API_KEY = 'sk-your-api-key'
RERANK_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-api/v1'
```

**最简调用测试**：
```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
completion = client.chat.completions.create(
    model="qwen3.8-max",
    messages=[{"role": "user", "content": "你是什么模型？"}]
)
print(completion.choices[0].message.content)
```

**Token用量说明**：
- `choices`：模型生成的候选回复列表
- `usage`：token用量统计
  - `prompt_tokens`：输入（提示）的token总数
  - `completion_tokens`：输出（生成）的token总数

**Embedding测试**：
```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

input = ["你是什么模型？", "将多模态输入编码为1个向量"] * 10

client = OpenAI(
    api_key=os.environ.get('api_key'),
    base_url=os.environ.get('base_url')
)

completion = client.embeddings.create(
    model="qwen3.7-text-embedding",
    input=input,
    dimensions=2048
)

for item in completion.data:
    print(item.embedding[:10])
    print(len(item.embedding))
```

**Rerank测试**：
```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url=os.getenv('RERANK_BASE_URL')
)

results = client.post(
    "/reranks",
    body={
        "model": "qwen3-rerank",
        "query": "什么是重排序模型",
        "documents": [
            "重排序模型广泛应用于搜索引擎和推荐系统",
            "量子计算是计算科学的前沿领域",
            "预训练语言模型为重排序模型带来新突破"
        ],
        "top_n": 3
    },
    cast_to=object
)
```

---

## 三、LangChain

### 3.1 LangChain简介

- **LangChain**：大模型应用开发框架，负责编排 LLM + Prompt + Memory + Chain + Indexes + Agent
- 核心价值：把大模型、提示词、记忆、链、索引、工具按统一接口组装
- 六大组件：Models、Prompts、Memory、Chains、Indexes、Agents

### 3.2 Models组件

- **Models**：统一封装大模型调用接口
- 支持OpenAI兼容接口（千问、Ollama等）
- `ChatOpenAI`：对话模型，接收消息列表，返回AI回复

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

client = ChatOpenAI(model='qwen3.8-max')
response = client.invoke("你是什么模型？")
print(response.content)
```

### 3.3 Prompts组件

- **Prompts**：模板化提示词管理
- `PromptTemplate`：基础模板，变量占位
- `FewShotPromptTemplate`：少样本模板，提供示例引导模型输出格式

**少样本信息抽取示例**：
```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

client = ChatOpenAI(model='qwen3.8-max')

examples = [
    {
        'text': '张三，28岁，在北京工作',
        'json': '{"name": "张三", "age": 28, "city": "北京"}',
    },
    {
        'text': '李四今年35岁，家住上海',
        'json': '{"name": "李四", "age": 35, "city": "上海"}',
    },
]

example_prompt = PromptTemplate(
    input_variables=['text', 'json'],
    template='文本: {text}\nJSON: {{{json}}}',
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix='从文本中抽取姓名、年龄、城市，只返回 JSON，不要其他说明。',
    suffix='文本: {input}',
    input_variables=['input'],
    example_separator='\n\n',
)

prompt_text = few_shot_prompt.format(input='王五，42岁，目前在广州定居')
response = client.invoke(prompt_text)
print(response.content)
```

### 3.4 Memory组件

- **Memory**：对话记忆，维护历史消息
- OpenAI SDK原生方式：维护messages列表，程序重启后从JSON文件恢复

**保存对话历史**：
```python
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI()
model = 'qwen3.8-max'

messages = [
    {'role': 'system', 'content': '你是用户的助手，友好、简洁地回答。'},
]

messages.append({'role': 'user', 'content': '你是我的助手，我给你起个名字叫贾维斯'})

response = client.chat.completions.create(model=model, messages=messages)
answer = response.choices[0].message.content
print(f'AI: {answer}')
messages.append({'role': 'assistant', 'content': answer})

with open('history.json', 'w', encoding='utf-8') as f:
    json.dump(messages, f, indent=2, ensure_ascii=False)
```

**加载对话历史**：
```python
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI()
model = 'qwen3.8-max'

with open('history.json', 'r', encoding='utf-8') as f:
    messages = json.load(f)

# 模拟程序重启：历史从文件读回，再追问
messages.append({'role': 'user', 'content': '你叫什么名字？'})
response = client.chat.completions.create(model=model, messages=messages)
answer = response.choices[0].message.content
print(f'AI: {answer}')
```

### 3.5 Chains组件

- **Chains**：链式编排，将多个步骤串联
- LangChain使用 Runnable 协议（`|` 管道符）连接组件
- `RunnablePassthrough`：透传输入，可 `.assign()` 追加字段
- `StrOutputParser`：提取模型输出的纯文本

**顺序链示例（生成标题→选出最佳标题）**：
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
client = ChatOpenAI(model='qwen3.8-max')

# 第1步：生成10个标题
titles_prompt = ChatPromptTemplate.from_messages([
    ('system', '根据关键词生成 10 个公众号文章标题，每行一个，只输出标题列表。'),
    ('user', '关键词: {keyword}'),
])
titles_chain = titles_prompt | client | StrOutputParser()

# 第2步：选出最好的标题并说明理由
pick_prompt = ChatPromptTemplate.from_messages([
    ('system', '从以下标题中选出最好的 1 个，并简要说明理由（2~3 句话）。'),
    ('user', '{titles}'),
])
pick_chain = pick_prompt | client | StrOutputParser()

# 在原输入字典上「追加」字段 titles
add_titles = RunnablePassthrough.assign(titles=titles_chain)
pipeline = add_titles | pick_chain

result = pipeline.invoke({'keyword': '武汉一日游'})
print(result)
```

**LLM请求链（抓取网页→根据正文回答问题）**：
```python
import requests
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
client = ChatOpenAI(model='qwen3.8-max', temperature=0)

def fetch_text(url: str) -> str:
    resp = requests.get(url, verify=False, timeout=10)
    resp.raise_for_status()
    text = BeautifulSoup(resp.text, 'html.parser').get_text(
        separator='\n', strip=True
    )
    return text[:4000]

prompt = ChatPromptTemplate.from_messages([
    ('system', '根据网页正文回答问题；正文中没有相关信息时回答「未找到」。'),
    ('user', '网页正文：\n>>> {requests_result} <<<\n\n问题：{query}'),
])

fetch_step = RunnablePassthrough.assign(
    requests_result=lambda x: fetch_text(x['url'])
)

chain = fetch_step | prompt | client | StrOutputParser()

result = chain.invoke({
    'url': 'https://wttr.in/武汉',
    'query': '武汉今天天气怎么样？',
})
print(result)
```

### 3.6 Indexes组件

- **Indexes**：文档处理流水线 Load → Split → Embed → Store → Retrieve
- **Load**：加载文档（txt/md/csv/pdf/docx等）
- **Split**：切分文档块（RecursiveCharacterTextSplitter）
- **Embed**：向量化文档块
- **Store**：存入向量数据库（Milvus）
- **Retrieve**：按相似度检索TopK

### 3.7 Agents组件

- **Agents**：大模型自主决策调用工具
- ReAct模式：Thought → Action → Observation 循环
- `@tool` 装饰器：将函数注册为工具
- `create_agent`：创建Agent实例

**手写ReAct Agent**：
```python
def run_react_agent(topic, max_steps=5):
    system_prompt = """你是公众号文章写作助手。
请按以下格式回答：

Thought: 思考下一步做什么
Action: 工具名
Action Input: 工具输入
...
Final Answer: 最终文章

可用工具：
1. web_search(query) - 搜索网页
2. fetch_webpage(url) - 抓取网页正文"""

    history = f"Thought: 我需要写一篇关于「{topic}」的文章\n"
    for _ in range(max_steps):
        output = llm(history)
        parsed = parse_output(output)
        if parsed["type"] == "final":
            return parsed["answer"]
        if parsed["type"] == "action":
            observation = run_tool(parsed["action"], parsed["input"])
            history += output + f"\nObservation: {observation}\nThought:"
            continue
        history += output + "\n请按格式继续。\nThought:"
    return "超过最大步数"
```

**LangChain @tool + create_agent**：
```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """根据关键词搜索网页，返回标题、链接和摘要。"""
    return _web_search(query)

@tool
def fetch_webpage(url: str) -> str:
    """打开指定 URL，提取网页正文内容。"""
    return _fetch_webpage(url)
```

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="qwen3.8-max", temperature=0)
tools = [web_search, fetch_webpage]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是公众号文章写作助手。先搜索、阅读资料，再撰写文章。",
)

response = agent.invoke({
    "messages": [HumanMessage(content="写一篇武汉一日游攻略的公众号文章")]
})
print(response["messages"][-1].content)
```

---

## 四、基础知识

### 4.1 Milvus向量数据库

**核心概念**：向量数据库专门存储嵌入向量，并按相似度做TopK检索。Milvus开源、支持持久化、按字段过滤、服务端部署。

**向量库与关系库对比**：

| 对比项 | 关系型数据库（MySQL） | 向量数据库（Milvus） |
|--------|----------------------|---------------------|
| 存什么 | 结构化行数据 | 向量 + 文本块 + 元数据 |
| 怎么查 | 等值、范围精确匹配 | 按语义相似度TopK召回 |
| 典型用途 | FAQ问答、业务表单 | RAG知识库检索 |
| 类比 | 表、行、列 | Collection、Entity、Field |

**Schema设计**：
```
Collection: demo_knowledge
├── id VARCHAR 主键（chunk 唯一标识）
├── text VARCHAR 切块正文（检索后拼进 context）
├── source VARCHAR 来源文件/分类（用于 filter）
└── vector FLOAT_VECTOR Embedding 向量（dim 与模型一致）
```

**连接**：
```python
from pymilvus import MilvusClient

# 课堂：Milvus Lite，数据落在本地 .db 文件
client = MilvusClient(uri="milvus_demo.db")

# 生产：连接 Milvus 服务
client = MilvusClient(uri="http://localhost:19530", db_name="default")

# 查看所有集合
print(client.list_collections())

# 查询数据
result = client.query(collection_name="edu_rag", limit=2, output_fields=["text", "source"])
```

**创建集合**：
```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", db_name="default")
COLLECTION = "demo_knowledge"
DIM = 4  # 课堂演示；真实场景改为 Embedding 模型输出维度

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="",
    index_name="vector_index",
)
client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
```

**插入数据**：
```python
chunks = [
    {
        "id": "chunk_rag_001",
        "text": "RAG 是检索增强生成：先检索相关知识，再交给大模型生成答案。",
        "source": "rag_intro.txt",
        "vector": [0.12, 0.85, 0.03, 0.44],
    },
    # ...
]
res = client.insert(collection_name=COLLECTION, data=chunks)
```

**向量搜索**：
```python
query_vector = [[0.10, 0.80, 0.08, 0.48]]

res = client.search(
    collection_name=COLLECTION,
    data=query_vector,
    limit=3,
    search_params={"metric_type": "COSINE"},
    output_fields=["text", "source"],
)

for hit in res[0]:
    print(hit["entity"]["text"], hit["distance"])
```

**带过滤搜索**：
```python
res = client.search(
    collection_name=COLLECTION,
    data=query_vector,
    limit=3,
    search_params={"metric_type": "COSINE"},
    output_fields=["text", "source"],
    filter='source == "rag_intro.txt"',
)
```

**更新与删除**：
```python
# upsert
client.upsert(collection_name=COLLECTION, data=[{
    "id": "chunk_rag_001",
    "text": "更新后的内容",
    "source": "rag_intro.txt",
    "vector": [0.15, 0.82, 0.05, 0.40],
}])

# 按主键删除
client.delete(collection_name=COLLECTION, ids=["chunk_rag_001"])

# 按条件删除
client.delete(collection_name=COLLECTION, filter='source == "rag_intro.txt"')
```

**度量方式**：
- **COSINE**：余弦相似度，方向越接近越相似，RAG常用
- **IP（内积）**：向量模长也影响结果，适合已归一化向量
- **L2**：欧氏距离，值越小越相似

**索引类型**：
- **FLAT**：暴力精确，适合小数据
- **IVF_***：先聚类再搜，大规模常用
- **HNSW**：图索引，低延迟场景常见

### 4.2 MySQL与ORM

**核心概念**：MySQL负责存结构化数据（FAQ问答对、会话消息、业务配置等），通过SQL做精确查询。

**FAQ表设计**：
```sql
CREATE TABLE IF NOT EXISTS faq_demo_tb (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(20),
    question VARCHAR(1000),
    answer VARCHAR(1000)
);
```

**PyMySQL CRUD**：

```python
import pymysql

connection = pymysql.connect(
    host="localhost", port=3306, user="edu_rag",
    password="app123456", database="edu_rag_project", charset="utf8mb4",
)
cursor = connection.cursor()

# 增
sql = "INSERT INTO faq_demo_tb (category, question, answer) VALUES (%s, %s, %s)"
cursor.execute(sql, ("RAG", "什么是 RAG？", "RAG 是检索增强生成，先检索再生成。"))
# 批量
cursor.executemany(sql, [("RAG", "问题1", "答案1"), ("LangChain", "问题2", "答案2")])
connection.commit()

# 查
cursor.execute("SELECT question FROM faq_demo_tb")
print(cursor.fetchall())
cursor.execute("SELECT answer FROM faq_demo_tb WHERE question=%s", ("什么是 RAG？",))
result = cursor.fetchone()

# 改
cursor.execute("UPDATE faq_demo_tb SET answer=%s WHERE id=%s", ("新答案", 1))
connection.commit()

# 删
cursor.execute("DELETE FROM faq_demo_tb WHERE id=%s", (3,))
connection.commit()
```

**SQLAlchemy ORM**：

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class FaqDemo(Base):
    __tablename__ = "faq_demo"
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(20))
    question = Column(String(1000))
    answer = Column(String(1000))
```

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = "mysql+pymysql://edu_rag:app123456@localhost:3306/edu_rag_project?charset=utf8mb4"
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)

with Session(engine) as session:
    # 增
    session.add(FaqDemo(category="RAG", question="什么是RAG？", answer="..."))
    session.commit()
    # 查
    rows = session.query(FaqDemo).all()
    row = session.query(FaqDemo).filter(FaqDemo.question == "什么是RAG？").first()
    # 改
    row.answer = "新答案"
    session.commit()
    # 删
    session.delete(row)
    session.commit()
```

### 4.3 TF-IDF算法

**核心概念**：TF-IDF（词频-逆文档频率）衡量「词对文档的重要性」：在本文常见，但在全库不常见的词，得分更高。

**原理**：
- **TF（Term Frequency，词频）**：词在本文档中出现越多，越重要
- **IDF（Inverse Document Frequency，逆文档频率）**：词在越多文档中出现，区分度越低，权重越小
- **TF-IDF(t, d) = TF(t, d) × IDF(t)**

**公式**：
```
TF(t, d) = 词t在文档d中的出现次数 / 文档d的总词数
IDF(t) = log(N / df(t))
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**纯原生实现**：
```python
import math
from collections import Counter
import jieba

documents = ["我喜欢编程", "编程真的很有趣", "音乐很美妙"]
tokenized_docs = [jieba.lcut(doc) for doc in documents]
N = len(tokenized_docs)

# 文档频率
df = Counter()
for tokens in tokenized_docs:
    for t in set(tokens):
        df[t] += 1

def idf(t):
    return math.log(N / df[t])

def tfidf_map(tokens):
    tf = Counter(tokens)
    length = len(tokens)
    return {t: (cnt / length) * idf(t) for t, cnt in tf.items() if t in df}

doc_vectors = [tfidf_map(tokens) for tokens in tokenized_docs]
```

**余弦相似度检索**：
```python
import numpy as np

vocab = sorted(df.keys())

def to_array(vec):
    return np.array([vec.get(t, 0.0) for t in vocab], dtype=float)

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

query = "他喜欢编程"
query_vec = to_array(tfidf_map(jieba.lcut(query)))
scores = [cosine(query_vec, to_array(v)) for v in doc_vectors]
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
```

**sklearn实现**：
```python
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = ["我喜欢编程", "编程真的很有趣", "音乐很美妙"]
tokenized_docs = [jieba.lcut(doc) for doc in documents]
corpus = [" ".join(tokens) for tokens in tokenized_docs]

vectorizer = TfidfVectorizer(analyzer=str.split)
doc_vectors = vectorizer.fit_transform(corpus)

query = "他喜欢编程"
query_vector = vectorizer.transform([" ".join(jieba.lcut(query))])
scores = cosine_similarity(query_vector, doc_vectors).ravel()
```

**TF-IDF vs 向量检索**：

| 对比项 | TF-IDF（关键词） | 向量检索（Milvus） |
|--------|-----------------|-------------------|
| 匹配方式 | 词项重叠 + 统计打分 | Embedding语义相似度 |
| 依赖 | 分词即可 | 向量模型 + 向量库 |
| 擅长 | 关键词、专有名词、字面相近 | 同义改写、语义相近表达 |
| 用途 | FAQ关键词打分 | 知识库语义TopK（RAG） |

---

## 五、项目初始化

### 5.1 FastAPI模板渲染

**核心概念**：FastAPI是现代高性能Python Web框架；uvicorn是ASGI服务器；Jinja2是模板引擎。

**基础结构**：
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def index():
    return "Hello, World!"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8089, reload=True)
```

**模板渲染**：
```python
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "EduRAG 智能问答",
            "message": "你好！我是智能问答助手，请输入你的问题。",
        },
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8089, reload=True)
```

**路由拆分**：
```python
# routes.py
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "EduRAG 智能问答", "message": "你好！我是智能问答助手。"},
    )

@router.get("/health")
def health():
    return {"status": "success"}
```

```python
# app.py
from fastapi import FastAPI
import uvicorn
from routes import router

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8089, reload=True)
```

### 5.2 Migrate与CLI

#### 5.2.1 配置管理

```python
# config.py
import os
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

class Config:
    def __init__(self):
        self.MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
        self.MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        self.MYSQL_USER = os.getenv("MYSQL_USER", "root")
        self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root123456")
        self.MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "edu_rag_project")

    @property
    def DATABASE_URL(self):
        user = quote_plus(self.MYSQL_USER)
        password = quote_plus(self.MYSQL_PASSWORD)
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

config = Config()
```

#### 5.2.2 数据库模型

```python
# db/base.py
from sqlalchemy.orm import declarative_base
Base = declarative_base()

# db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import config

engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# db/models.py
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Text
from db.base import Base

class Faq(Base):
    __tablename__ = "faq"
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=True)
    question = Column(String(1000), nullable=True)
    answer = Column(String(2000), nullable=True)
    tokens = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)

class RagChunk(Base):
    __tablename__ = "rag_chunk"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, nullable=True)
    doc_file_name = Column(String(256), nullable=True)
    doc_index = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    source = Column(String(50), nullable=True)
```

#### 5.2.3 Alembic迁移

**env.py关键配置**：
```python
from db.base import Base
from config import config as app_config
import db.models

config.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)
target_metadata = Base.metadata
```

**常用命令**：

| 命令 | 作用 |
|------|------|
| `alembic revision --autogenerate -m "create faq"` | 生成迁移脚本 |
| `alembic upgrade head` | 升级到最新版本 |
| `alembic current` | 查看当前版本 |
| `alembic history` | 查看迁移历史 |
| `alembic downgrade -1` | 回退一个版本 |

#### 5.2.4 自研CLI

```python
# cli.py
import argparse
import importlib
import pkgutil

def _discover_commands():
    """扫描 commands 目录，加载含 NAME / register / run 的模块。"""
    import commands as commands_pkg
    found = []
    for info in pkgutil.iter_modules(commands_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"commands.{info.name}")
        except ImportError as e:
            print(f"[cli] 跳过命令模块 commands.{info.name}: {e}")
            continue
        if (
            getattr(module, "NAME", None)
            and callable(getattr(module, "register", None))
            and callable(getattr(module, "run", None))
        ):
            found.append(module)
    return sorted(found, key=lambda m: m.NAME)

def main(argv=None):
    parser = argparse.ArgumentParser(description="EduRAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    handlers = {}
    for mod in _discover_commands():
        mod.register(sub)
        handlers[mod.NAME] = mod.run
    args = parser.parse_args(argv)
    handlers[args.command](args)

if __name__ == "__main__":
    main()
```

**命令模块示例**：
```python
# commands/demo.py
NAME = "demo"
HELP = "打印一行，验证 commands 是否通"

def register(sub):
    parser = sub.add_parser(NAME, help=HELP)
    parser.add_argument("--question", help="用户问题")

def run(args):
    print("OK: commands 已连通，可以开始写业务命令了")
    if args.question:
        print("用户问题", args.question)
```

### 5.3 建Milvus集合

**.env追加配置**：
```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DATABASE_NAME=default
MILVUS_COLLECTION_NAME=edu_rag
EMBEDDING_DIM=1024
```

**config.py追加**：
```python
self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
self.MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
self.MILVUS_DATABASE_NAME = os.getenv("MILVUS_DATABASE_NAME", "default")
self.MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "default")
self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
```

**Milvus客户端**：
```python
# clients/milvus.py
from pymilvus import MilvusClient
from config import config

class MilvusDBClient:
    def __init__(self, host=None, port=None, database=None):
        host = host or config.MILVUS_HOST
        port = port or config.MILVUS_PORT
        database = database or config.MILVUS_DATABASE_NAME
        self.uri = f"http://{host}:{port}"
        self.database = database
        self.client = MilvusClient(uri=self.uri, db_name=database)

    def list_collections(self):
        return self.client.list_collections()

    def has_collection(self, name):
        return self.client.has_collection(name)
```

**init-milvus命令**：
```python
# commands/init_milvus.py
from pymilvus import DataType
from clients.milvus import MilvusDBClient
from config import config

NAME = "init-milvus"
HELP = "创建 Milvus 集合（若不存在）"

def create_collection(milvus, name=None):
    name = name or config.MILVUS_COLLECTION_NAME
    if milvus.has_collection(name):
        milvus.client.load_collection(name)
        return

    schema = milvus.client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    schema.add_field("file_name", DataType.VARCHAR, max_length=256)
    schema.add_field("index", DataType.INT64)
    schema.add_field("parent_id", DataType.INT64)

    index_params = milvus.client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    milvus.client.create_collection(
        collection_name=name, schema=schema, index_params=index_params
    )
    milvus.client.load_collection(name)

def run(args):
    milvus = MilvusDBClient()
    create_collection(milvus)
```

### 5.4 大模型接入

**.env追加**：
```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your-api-key
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v3
```

**LLM客户端**：
```python
# clients/llm.py
from openai import OpenAI
from config import config

class LLMClient:
    def __init__(self):
        api_key = config.OPENAI_API_KEY
        self.client = OpenAI(api_key=api_key, base_url=config.OPENAI_BASE_URL)
        self.model = config.LLM_MODEL

    def chat(self, messages, temperature=0.3):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=60,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"抱歉，模型调用失败：{e}"

    def chat_stream(self, messages, temperature=0.3):
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=60,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"抱歉，模型调用失败：{e}"
```

**Embedding客户端**：
```python
# clients/embedding.py
from openai import OpenAI
from config import config

class EmbeddingClient:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        self.model = config.EMBEDDING_MODEL
        self.dim = config.EMBEDDING_DIM

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return []
        all_vectors = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            resp = self.client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            vectors = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
            all_vectors.extend(vectors)
        return all_vectors
```

### 5.5 接入Logger

**核心概念**：Python标准库`logging`无需额外安装。日志级别：DEBUG → INFO → WARNING → ERROR → CRITICAL。三要素：Logger（记录器）、Handler（输出目标）、Formatter（输出格式）。

**.env追加**：
```env
LOG_FILE=storage/logs/app.log
```

```python
# logger.py
import logging
import os
from config import config

def setup_logger(name=None, log_file=None):
    log_file = str(log_file or config.LOG_FILE)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)

    if lg.handlers:
        return lg

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    lg.addHandler(console_handler)
    lg.addHandler(file_handler)
    return lg

logger = setup_logger()
```

---

## 六、FAQ问答

### 6.1 Greeting（正则寒暄短路）

**核心概念**：问候语走短路处理，不进入后续流程。关键词整句匹配（去标点），避免「你好，1+1等于几？」误命中。

```python
# services/greeting.py
import random

GREETING_KEYWORDS = [
    "你好", "您好", "hi", "hello", "你是谁", "您是谁",
    "你叫什么", "你的名字", "who are you", "在吗", "在不在", "有人吗",
]

GREETING_RESPONSES = [
    "你好！我是传智教育智能助教，有什么可以帮你？",
    "传智教育智能助手已就位，小主有何吩咐？",
    "学IT就来黑马程序员，有什么可以帮你？",
]

def match_greeting(query: str):
    text = (query or "").strip().lower()
    for ch in "？?！!。.，,":
        text = text.replace(ch, "")
    text = text.strip()
    if text in {k.lower() for k in GREETING_KEYWORDS}:
        return random.choice(GREETING_RESPONSES)
    return None
```

### 6.2 导入FAQ（CSV → MySQL）

**核心概念**：FAQ CSV约66条，三列：`分类,问题,答案`。导入时分词，写入同表`tokens`列。

**文本预处理**：
```python
# utils/preprocess.py
import jieba
from pathlib import Path

def _get_stopwords():
    stopwords_path = Path(__file__).parent / "../storage/data/cn_stopwords.txt"
    with open(stopwords_path, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

STOPWORDS = _get_stopwords()

def preprocess_text(text: str):
    text = text.lower()
    return [word for word in jieba.lcut(text) if word not in STOPWORDS and word.strip()]
```

**FAQ导入命令**：
```python
# commands/ingest_faq.py
import pandas as pd
from pathlib import Path
import json
from db.models import Faq
from db.session import SessionLocal
from utils.preprocess import preprocess_text

NAME = "ingest-faq"
HELP = "导入 FAQ 数据到 MySQL"

def write_to_mysql(data):
    with SessionLocal() as session:
        for row in data.to_dict(orient="records"):
            question = row["问题"]
            if not question or not str(question).strip():
                continue
            has = session.query(Faq).filter(Faq.question == question).first()
            if has:
                continue
            tokens = json.dumps(preprocess_text(row["问题"]), ensure_ascii=False)
            faq = Faq(
                category=row["分类"],
                question=question,
                answer=row["答案"],
                tokens=tokens,
            )
            session.add(faq)
        session.commit()

def run(args):
    csv_path = Path(__file__).parent / "../storage/data/课程咨询FAQ.csv"
    data = pd.read_csv(csv_path)
    write_to_mysql(data)
```

### 6.3 TF-IDF检索

**核心概念**：TF-IDF + 余弦相似度做关键词检索。余弦相似度落在0~1，用`TFIDF_THRESHOLD`判断是否命中。启动时加载分词建索引，查询时只对用户问题分词。

**阈值配置**：
```env
TFIDF_THRESHOLD=0.5
```

**TF-IDF搜索引擎**：
```python
# services/tfidf.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from logger import logger

class TfidfSearch:
    def __init__(self, tokenized_docs):
        corpus = [" ".join(doc) for doc in tokenized_docs]
        self.vectorizer = TfidfVectorizer(analyzer=str.split)
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        logger.info(f"TfidfVectorizer 初始化完成，共 {len(tokenized_docs)} 条")

    def get_scores(self, tokenized_query):
        q = " ".join(tokenized_query)
        q_vec = self.vectorizer.transform([q])
        return cosine_similarity(q_vec, self.doc_vectors).ravel()
```

**FAQ服务**：
```python
# services/faq.py
import json
import numpy as np
from config import config
from db.models import Faq
from db.session import SessionLocal
from logger import logger
from services.tfidf import TfidfSearch
from utils.preprocess import preprocess_text

class FaqService:
    def __init__(self):
        self.ids = []
        self.questions = []
        self.answers = []
        self.tfidf = None
        self._reload()

    def _reload(self):
        with SessionLocal() as session:
            db_rows = session.query(Faq).all()
        self.ids, self.questions, self.answers, tokenized = [], [], [], []
        for row in db_rows:
            if not row.tokens:
                continue
            tokens = json.loads(row.tokens)
            if not tokens:
                continue
            self.ids.append(row.id)
            self.questions.append(row.question)
            self.answers.append(row.answer)
            tokenized.append(tokens)
        if not tokenized:
            self.tfidf = None
            return
        self.tfidf = TfidfSearch(tokenized)

    def search(self, query, threshold=None):
        threshold = config.TFIDF_THRESHOLD if threshold is None else threshold
        if not query or not self.tfidf:
            return None, 0.0
        scores = self.tfidf.get_scores(preprocess_text(query))
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < threshold:
            return None, best_score
        return self.answers[best_idx], best_score
```

### 6.4 接口封装

**核心概念**：把Greeting → TF-IDF → miss抽到`QAPipeline.query`，暴露为`POST /api/ask`。

**问答Pipeline**：
```python
# pipeline.py
from logger import logger
from services.faq import FaqService
from services.greeting import match_greeting

class QAPipeline:
    def __init__(self):
        self.faq_service = FaqService()

    def query(self, question):
        question = (question or "").strip()
        logger.info(f"查询: {question!r}")

        answer = match_greeting(question)
        if answer:
            return {"answer": answer, "stage": "greeting", "score": 1.0}

        answer, score = self.faq_service.search(question)
        if answer:
            return {"answer": answer, "stage": "faq", "score": score}

        return {"answer": "暂未在 FAQ 中找到可靠答案。", "stage": "miss", "score": score}
```

**API接口**：
```python
# routes.py
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

@router.post("/api/ask")
def ask(body: QueryRequest):
    pipeline = QAPipeline()
    return pipeline.query(body.question)
```

**前端交互**：
```javascript
async function askBackend(session, question) {
    const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
    });
    const data = await resp.json();
    session.messages.push({
        role: "bot",
        text: `${data.answer}${data.stage ? `\n\n[${data.stage}]` : ""}`,
    });
}
```

---

## 七、RAG知识库

### 7.1 文档预处理

**核心概念**：入库前先把PDF/扫描件/图片整理成干净文本或Markdown。

- **MinerU**：PDF/Office/图片 → 结构化Markdown（含表格、阅读顺序），依赖重
- **RapidOCR**：图片/扫描页快速出字，轻量，onnxruntime即可跑
- **推荐流水线**：原始文档 → 预处理 → .md/.txt → 字符切分 → 父子分块 → 入库

**RapidOCR单张图片**：
```python
from pathlib import Path
from rapidocr import RapidOCR

engine = RapidOCR()
result = engine("./scan_page.png")
lines = [t for t in result.txts if t] if result and getattr(result, "txts", None) else []
text = "\n".join(lines)
Path("./scan_page.txt").write_text(text, encoding="utf-8")
```

**RapidOCR PDF**：
```python
from pathlib import Path
from rapidocr_pdf import RapidOCRPDF

extractor = RapidOCRPDF()
pages = extractor("./AI大模型开发8.0简版大纲.pdf", force_ocr=False, page_num_list=[0, 1, 2])

chunks = [f"## 第 {page_no + 1} 页\n{text}" for page_no, text, score in pages]
body = "\n\n".join(chunks)
Path("./AI大模型开发8.0简版大纲.txt").write_text(body, encoding="utf-8")
```

### 7.2 字符切分器

**核心概念**：
- `chunk_size`：每个块的字符数
- `chunk_overlap`：重叠字符数，解决语义在chunk边界被切断的问题
- 下一窗起点 = 上一窗终点 - overlap；overlap必须小于chunk_size
- **RecursiveCharacterTextSplitter**：按"段落 → 句子 → 词 → 字符"的优先级递归切分

**滑动窗口切分器**：
```python
def split_by_chars(text, chunk_size, chunk_overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - chunk_overlap
    return chunks
```

**RecursiveCharacterTextSplitter对比**：
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=8,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)
splits = splitter.split_text(text)
```

### 7.3 文档加载和切分

**处理规则**：
- `.md/.txt`：直接读文本，按CHUNK_SIZE切块
- `.csv`：按表头把每一行拼成区块（列名：值，不二次切片）
- 其他（.pdf/.docx等）：先UnstructuredLoader.load，再按CHUNK_SIZE切块

**配置参数**：
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

**统一加载入口**：
```python
# utils/chunking.py
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config
from logger import logger

def load_text_and_chunk(file_path):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8").strip()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    splits = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(splits):
        docs.append(Document(
            page_content=chunk,
            metadata={"file_name": file_path.name, "index": i},
        ))
    return docs

def load_csv_and_chunk(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    docs = []
    for index, row in df.iterrows():
        chunk = '\n'.join([f'{col}: {row[col]}' for col in df.columns])
        docs.append(Document(
            page_content=chunk,
            metadata={"file_name": file_path.name, "index": index},
        ))
    return docs

def load_and_chunk(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix in [".md", ".txt"]:
        return load_text_and_chunk(file_path)
    if suffix == ".csv":
        return load_csv_and_chunk(file_path)
    else:
        return load_file_and_chunk(file_path)  # UnstructuredLoader
```

### 7.4 父子分块与入库

**核心概念**：**父子分块（Parent-Child Chunking）**——检索用小块，生成用大块。
- 子块入Milvus（向量 + 子块text + parent_id）做细粒度召回
- 父块入MySQL（原文），检索时按parent_id回表取父块
- CSV一行已是完整问答 → 1父1子（正文相同）

**配置参数**：
```env
CHILD_CHUNK_SIZE=200
CHILD_CHUNK_OVERLAP=20
```

**父块入库（MySQL）**：
```python
# services/chunk_store.py
from db.models import RagChunk
from db.session import SessionLocal
from logger import logger

class ChunkStore:
    def add_chunks(self, parents):
        if not parents:
            return []
        with SessionLocal() as session:
            for doc in parents:
                doc_file_name = doc.metadata.get("file_name") or ""
                doc_index = int(doc.metadata.get("index", 0))
                row = session.query(RagChunk).filter(
                    RagChunk.doc_file_name == doc_file_name,
                    RagChunk.doc_index == doc_index,
                ).first()
                if not row:
                    row = RagChunk(
                        doc_file_name=doc_file_name,
                        doc_index=doc_index,
                        content=doc.page_content,
                    )
                    session.add(row)
                    session.flush()
                doc.metadata["id"] = row.id
            session.commit()
            return parents
```

**子块入库（Milvus）**：
```python
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from clients.embedding import EmbeddingClient
from clients.milvus import MilvusDBClient
from config import config

class ChunkStore:
    def __init__(self):
        self.embedding = EmbeddingClient()
        self.milvus = MilvusDBClient()

    def add_child_chunks(self, parents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        all_children = []
        for parent in parents:
            suffix = Path(parent.metadata["file_name"]).suffix.lower()
            if suffix == ".csv":
                child_chunks = [parent.page_content]
            else:
                child_chunks = splitter.split_text(parent.page_content)
            if not child_chunks:
                continue
            vectors = self.embedding.embed(child_chunks)
            data = []
            for text, vector in zip(child_chunks, vectors):
                data.append({
                    "id": hashlib.md5(text.encode("utf-8")).hexdigest(),
                    "text": text,
                    "vector": vector,
                    "file_name": parent.metadata["file_name"],
                    "index": parent.metadata["index"],
                    "parent_id": parent.metadata["id"],
                })
            self.milvus.client.insert(
                collection_name=config.MILVUS_COLLECTION_NAME,
                data=data,
            )
            all_children.extend(data)
        return all_children
```

**ingest-rag命令**：
```python
# commands/ingest_rag.py
NAME = "ingest-rag"
HELP = "RAG 知识库入库"

def run(args):
    path = Path(args.file_name)
    chunk_store = ChunkStore()
    parents = load_and_chunk(path)
    parents = chunk_store.add_chunks(parents)
    children = chunk_store.add_child_chunks(parents)
    # ES双写
    EsStore().add_children(children)
```

### 7.5 向量检索与阈值

**核心概念**：
- 子块候选数 = RETRIEVAL_K * 4（宁多勿少）
- Milvus Range Search：COSINE下 `radius < score ≤ range_filter`
- 按parent_id去重（同一父下多子块只留最高分）
- MySQL回表取父块原文

**配置参数**：
```env
RETRIEVAL_K=5
RAG_SCORE_THRESHOLD=0.45
```

**向量检索**：
```python
# services/rag.py
from config import config
from clients.embedding import EmbeddingClient
from clients.milvus import MilvusDBClient
from db.models import RagChunk
from db.session import SessionLocal
from logger import logger

class RAGService:
    def __init__(self):
        self.embedding = EmbeddingClient()
        self.milvus = MilvusDBClient()

    def vector_retrieval(self, question):
        vector = self.embedding.embed([question])[0]
        candidate_m = config.RETRIEVAL_K * 4
        results = self.milvus.client.search(
            collection_name=config.MILVUS_COLLECTION_NAME,
            data=[vector],
            anns_field="vector",
            search_params={
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 10,
                    "radius": config.RAG_SCORE_THRESHOLD,
                    "range_filter": 1.0,
                },
            },
            limit=candidate_m,
            output_fields=["text", "file_name", "index", "parent_id"],
        )[0]
        return results

    def retrieval(self, question):
        results = self.vector_retrieval(question)
        if not results:
            return [], 0.0
        ordered_results = {}
        for hit in results:
            entity = hit.get("entity") or hit
            score = float(hit.get("distance", 0.0))
            parent_id = entity.get("parent_id")
            if parent_id is None or parent_id in ordered_results:
                continue
            ordered_results[int(parent_id)] = score
        parent_ids = list(ordered_results.keys())[:config.RETRIEVAL_K]
        if not parent_ids:
            return [], 0.0
        with SessionLocal() as session:
            rows = session.query(RagChunk).filter(RagChunk.id.in_(parent_ids)).all()
        best_score = ordered_results[parent_ids[0]]
        return rows, best_score
```

### 7.6 Prompt生成与Pipeline

**RAG Prompt模板**：
```python
# prompts.py
RAG_PROMPT = """你是传智教育的智能助教。请基于「检索上下文」回答用户问题。
要求：
1. 优先依据上下文作答，不要编造上下文没有的细节；
2. 若上下文信息不足，明确说明不足；
3. 回答简洁、条理清晰。

检索上下文：
{context}

用户问题：{question}

回答："""
```

**生成方法**：
```python
class RAGService:
    def __init__(self):
        self.llm = LLMClient()

    def generate(self, question, rows):
        context = ''
        for row in rows:
            prefix = f"\n===== 《{row.doc_file_name}》第{row.doc_index}段，内容如下： ======\n"
            context += prefix + row.content
        prompt = RAG_PROMPT.format(context=context, question=question)
        result = self.llm.chat([{'role': 'user', 'content': prompt}])
        return result
```

**Pipeline接入RAG**：
```python
class QAPipeline:
    def __init__(self):
        self.faq_service = FaqService()
        self.rag = RAGService()

    def query(self, question):
        # greeting / faq 保持不变...
        rows, best_score = self.rag.retrieval(question)
        if rows:
            answer = self.rag.generate(question, rows)
            return {"answer": answer, "stage": "rag", "score": best_score}
        return {"answer": "暂未在知识库中找到可靠答案。", "stage": "miss", "score": best_score}
```

### 7.7 BM25算法和ES简介

**BM25原理**：保留TF-IDF的IDF思想，改进两点：
1. **词频饱和**：词频升高时得分增长变缓（避免高频词淹没其他信号）
2. **文档长度归一化**：相对平均文档长度做调节（避免长文档天然占优）

**BM25公式**：
```
fᵢ: 词项qᵢ在文档D中的出现次数（TF）
N: 文档总数
nᵢ: 包含词项qᵢ的文档数
|D|: 文档D的长度
avgdl: 所有文档的平均长度
k₁=1.2, b=0.75
```

**Elasticsearch**：开源分布式全文搜索引擎，倒排索引，默认BM25打分。

**三库分工**：

| 数据库 | 职责 |
|--------|------|
| MySQL | 存得住、查得准 |
| Milvus | 意思像不像 |
| ES | 字面词打不打得上 |

**rank_bm25库实现**：
```python
import jieba
from rank_bm25 import BM25L

class BM25Search:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [jieba.lcut(doc) for doc in documents]
        self.bm25 = BM25L(self.tokenized_docs)

    def search(self, query, top_k=1):
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc": self.documents[idx], "score": float(score), "index": idx}
                for idx, score in ranked]
```

### 7.8 ES关键词召回

**核心概念**：ES与Milvus同一粒度，每条子块一条ES文档。`_id` = 子块id（正文MD5）。存入`parent_id`供回表父块。`match`查询走ES默认BM25。

**Docker配置**：
```yaml
elasticsearch:
    container_name: elasticsearch
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - ./volumes/elasticsearch:/usr/share/elasticsearch/data
```

**配置参数**：
```env
ES_HOSTS=http://127.0.0.1:9200
ES_INDEX_NAME=edu_rag_child
ES_TOP_K=10
```

**ES客户端与存储**：
```python
# clients/elasticsearch.py
from elasticsearch import Elasticsearch
from config import config

class ElasticsearchClient:
    def __init__(self):
        self.client = Elasticsearch(config.ES_HOSTS.split(","))
        self.index = config.ES_INDEX_NAME

# services/es_store.py
INDEX_BODY = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "parent_id": {"type": "integer"},
            "doc_file_name": {"type": "keyword"},
            "doc_index": {"type": "integer"},
            "content": {"type": "text"},
        }
    },
}

class EsStore:
    def add_children(self, children):
        ops = []
        for row in children:
            chunk_id = str(row["id"])
            ops.append({"index": {"_index": self.es.index, "_id": chunk_id}})
            ops.append({
                "chunk_id": chunk_id,
                "parent_id": int(row["parent_id"]),
                "doc_file_name": row.get("file_name") or "",
                "doc_index": int(row.get("index") or 0),
                "content": row.get("text") or "",
            })
        self.es.client.bulk(operations=ops, refresh=True)

    def search(self, query, top_k=None):
        top_k = top_k or config.ES_TOP_K
        resp = self.es.client.search(
            index=self.es.index,
            size=top_k,
            query={"match": {"content": query}},
        )
        hits = []
        for h in resp.get("hits", {}).get("hits", []):
            src = h.get("_source") or {}
            hits.append({
                "chunk_id": str(src.get("chunk_id") or h["_id"]),
                "parent_id": int(src.get("parent_id") or 0),
                "score": float(h.get("_score") or 0.0),
                "content": src.get("content") or "",
            })
        return hits
```

### 7.9 多路召回与RRF

**核心概念**：
- **多路召回**：Milvus稠密（语义） ∥ ES稀疏（关键词）并行召回
- **RRF（Reciprocal Rank Fusion）**：按chunk_id去重融合，只看名次不看绝对分
- RRF公式：`score(d) = Σ 1 / (k + rank_i(d))`，常用 k=60

**稠密侧子块名次**：
```python
class RAGService:
    def dense_child_ranks(self, question, top_k=None):
        top_k = top_k or config.RETRIEVAL_K
        vector = self.embedding.embed([question])[0]
        candidate_m = top_k * 4
        results = self.milvus.client.search(
            collection_name=config.MILVUS_COLLECTION_NAME,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=candidate_m,
            output_fields=["text", "file_name", "index", "parent_id"],
        )[0]
        ordered = {}
        for hit in results:
            entity = hit.get("entity") or hit
            score = float(hit.get("distance", 0.0))
            chunk_id = str(hit.get("id") or entity.get("id") or "")
            parent_id = entity.get("parent_id")
            if not chunk_id or parent_id is None or chunk_id in ordered:
                continue
            ordered[chunk_id] = (score, int(parent_id))
        ranked = [(cid, sc, pid) for cid, (sc, pid) in list(ordered.items())[:top_k]]
        return ranked
```

**RRF融合**：
```python
# services/hybrid.py
def rrf_fuse(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for rank, chunk_id in enumerate(ranks, start=1):
            cid = str(chunk_id)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class HybridRetriever:
    def __init__(self, rag=None, es_store=None):
        self.rag = rag or RAGService()
        self.es_store = es_store or EsStore()

    def search(self, question, top_k=None):
        top_k = top_k or config.RETRIEVAL_K
        fetch_k = top_k * 2
        dense = self.rag.dense_child_ranks(question, top_k=fetch_k)
        sparse = self.es_store.search(question, top_k=fetch_k)
        parent_by_chunk = {}
        for cid, _, pid in dense:
            parent_by_chunk[cid] = pid
        for h in sparse:
            parent_by_chunk[h["chunk_id"]] = int(h["parent_id"])
        dense_ids = [cid for cid, _, _ in dense]
        sparse_ids = [h["chunk_id"] for h in sparse]
        fused_ids = rrf_fuse([dense_ids, sparse_ids])[:top_k]
        fused = [(cid, rrf, parent_by_chunk.get(cid))
                 for cid, rrf in fused_ids if parent_by_chunk.get(cid) is not None]
        dense_best = float(dense[0][1]) if dense else 0.0
        return fused, dense_best
```

### 7.10 重排与Pipeline接入

**核心概念**：
- **qwen3.7-text-rerank**：千问文本重排模型，精排排序
- 重排分relevance_score在0~1，适合本次候选内排序，不用它做hit门禁
- 是否命中仍用向量最高分 ≥ RAG_SCORE_THRESHOLD 判断
- 完整流程：Greeting → TF-IDF FAQ → RAG.retrieve（多路→RRF→向量阈值→升父块→重排→generate）

**配置参数**：
```env
RERANK_MODEL=qwen3.7-text-rerank
RERANK_TOP_N=5
```

**Rerank客户端**：
```python
# clients/rerank.py
from http import HTTPStatus
import dashscope
from dashscope import TextReRank
from config import config
from logger import logger

class RerankClient:
    def __init__(self, model=None):
        self.model = model or config.RERANK_MODEL

    def rerank(self, query, documents, top_n=None):
        if not documents:
            return []
        top_n = top_n if top_n is not None else config.RERANK_TOP_N
        resp = TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            instruct="Given a web search query, retrieve relevant passages that answer the query.",
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"rerank 失败: {resp.code} {resp.message}")
        results = (resp.output or {}).get("results") or []
        ranked = [
            {"index": int(item["index"]), "relevance_score": float(item["relevance_score"])}
            for item in results
        ]
        return ranked
```

**完整检索方法**：
```python
class RAGService:
    def __init__(self):
        self.embedding = EmbeddingClient()
        self.milvus = MilvusDBClient()
        self.llm = LLMClient()
        self.hybrid = HybridRetriever(rag=self)
        self.rerank = RerankClient()

    def retrieve(self, question, threshold=None):
        threshold = config.RAG_SCORE_THRESHOLD if threshold is None else threshold
        fused, dense_best = self.hybrid.search(question)
        if dense_best < threshold:
            return [], dense_best
        if not fused:
            return [], dense_best

        # 子块有序 → 父块去重（保留首次出现顺序）
        parent_ids, seen = [], set()
        for _cid, _rrf, pid in fused:
            pid = int(pid)
            if pid in seen:
                continue
            seen.add(pid)
            parent_ids.append(pid)

        with SessionLocal() as session:
            rows = session.query(RagChunk).filter(RagChunk.id.in_(parent_ids)).all()
        by_id = {int(r.id): r for r in rows}
        ordered_rows = [by_id[pid] for pid in parent_ids if pid in by_id]
        if not ordered_rows:
            return [], dense_best

        # 重排
        docs = [r.content or "" for r in ordered_rows]
        ranked = self.rerank.rerank(question, docs, top_n=len(docs))
        if not ranked:
            return ordered_rows[:config.RETRIEVAL_K], dense_best

        picked = []
        for item in ranked:
            picked.append(ordered_rows[item["index"]])
            if len(picked) >= config.RETRIEVAL_K:
                break
        return picked, dense_best
```

**完整检索流程图**：
```
Greeting → TF-IDF FAQ
  → RAG.retrieve:
       Milvus ∥ ES（子块）→ RRF(chunk_id)
       → 稠密 best_score ≥ RAG_SCORE_THRESHOLD？
            否 → miss（→ web）
            是 → 子块升父块（按序去重）→ MySQL → rerank排序
               → generate（stage=rag）
```

---

## 八、联网与兜底

### 8.1 SerpAPI与Bing搜索

**核心概念**：知识库未命中时，通过SerpAPI（Google搜索）或Bing搜索获取实时信息。

- **SerpAPI**：Google搜索API，返回结构化搜索结果（标题、链接、摘要）
- **requests/BeautifulSoup**：抓取网页正文详情
- **Selenium**：动态渲染页面抓取（备选方案）

**SerpAPI搜索**：
```python
from serpapi import GoogleSearch

params = {
    "engine": "google",
    "q": "查询内容",
    "api_key": "your-serpapi-key",
}
search = GoogleSearch(params)
results = search.get_dict()
organic_results = results.get("organic_results", [])
```

### 8.2 联网搜索接入

**核心概念**：SerpAPI搜索 + Prompt + Pipeline，stage=web。

```python
# services/web_search.py
from clients.llm import LLMClient
from config import config
from logger import logger

class WebSearchService:
    def __init__(self):
        self.llm = LLMClient()

    def answer(self, question):
        # 1. SerpAPI搜索
        results = self._search(question)
        if not results:
            return None, None
        # 2. 抓取详情
        snippets = [r.get("snippet", "") for r in results[:3]]
        # 3. Prompt生成
        context = "\n\n".join(snippets)
        prompt = WEB_PROMPT.format(context=context, question=question)
        answer = self.llm.chat([{"role": "user", "content": prompt}])
        return answer, "web"
```

**Pipeline接入**：
```python
class QAPipeline:
    def __init__(self):
        self.faq_service = FaqService()
        self.rag = RAGService()
        self.web = WebSearchService()
        self.fallback = FallbackService()

    def query(self, question, enable_rag=True, enable_web=True):
        # greeting / faq / rag 保持不变...
        if enable_web:
            answer, stage = self.web.answer(question)
            if stage == "web":
                return {"answer": answer, "stage": "web", "score": 1.0}
        # fallback
        answer = self.fallback.answer(question)
        return {"answer": answer, "stage": "fallback", "score": 1.0}
```

### 8.3 Fallback兜底

**核心概念**：所有阶段都未命中时的兜底回复，引导用户联系人工客服。

```python
# services/fallback.py
class FallbackService:
    def answer(self, question):
        return (
            "抱歉，我暂时无法回答您的问题。"
            "您可以通过以下方式联系我们：\n"
            "📞 客服电话：400-xxx-xxxx\n"
            "💬 在线客服：工作日 9:00-18:00"
        )
```

### 8.4 前端联调

**核心概念**：全链路验证——前端发送问题，后端返回答案+stage+score，前端渲染阶段标签。

**阶段标签配置**：
```javascript
const STAGE_META = {
    greeting: { label: "问候", color: "blue" },
    bm25:     { label: "FAQ",  color: "green" },
    rag:      { label: "知识库", color: "purple" },
    web:      { label: "联网", color: "orange" },
    fallback: { label: "兜底", color: "gray" },
};
```

---

## 九、项目工程化

### 9.1 流式输出（WebSocket + 打字机）

**核心概念**：通过WebSocket实现打字机效果，LLM流式输出token，前端逐字渲染。

```python
# routes.py
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/api/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            question = data.get("question", "").strip()
            if not question:
                continue
            pipeline = QAPipeline()
            # 非流式获取stage和score
            result = pipeline.query(question)
            # 流式输出answer
            for chunk in pipeline.rag.llm.chat_stream(
                [{"role": "user", "content": result["answer"]}]
            ):
                await ws.send_json({"type": "chunk", "content": chunk})
            await ws.send_json({
                "type": "end",
                "stage": result["stage"],
                "score": result["score"],
            })
    except WebSocketDisconnect:
        pass
```

**前端WebSocket**：
```javascript
const ws = new WebSocket(`ws://${location.host}/api/stream`);
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "chunk") {
        // 逐字追加
        botMessage.textContent += data.content;
    } else if (data.type === "end") {
        // 显示阶段标签
        showStageBadge(data.stage, data.score);
    }
};
ws.send(JSON.stringify({ question: userInput }));
```

### 9.2 建表与新建会话

**核心概念**：sessions表存储会话元数据，messages表存储每条消息。

**数据模型**：
```python
class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String(36), primary_key=True)  # UUID
    title = Column(String(200), nullable=False, default="新对话")
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    stage = Column(String(32), nullable=True)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
```

**接口规范**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/sessions` | POST | 新建会话 |
| `/api/sessions` | GET | 会话列表 |
| `/api/sessions/{id}` | GET | 单个会话 |
| `/api/sessions/{id}` | PATCH | 更新标题 |
| `/api/sessions/{id}` | DELETE | 删除会话 |
| `/api/sessions/{id}/messages` | GET | 消息列表 |

### 9.3 会话列表与消息

**核心概念**：前端侧边栏展示会话列表，支持切换、删除、标题重写。

**会话服务**：
```python
# services/conversation.py
class ConversationService:
    def create_session(self, title="新对话"):
        session_id = str(uuid.uuid4())
        with SessionLocal() as db:
            session = ChatSession(id=session_id, title=title)
            db.add(session)
            db.commit()
            return {"id": session_id, "title": title}

    def list_sessions(self):
        with SessionLocal() as db:
            rows = db.query(ChatSession).order_by(desc(ChatSession.updated_at)).all()
            return [{"id": s.id, "title": s.title} for s in rows]

    def get_messages(self, session_id):
        with SessionLocal() as db:
            rows = db.query(Message).filter(Message.session_id == session_id).all()
            return [{"role": m.role, "content": m.content, "stage": m.stage} for m in rows]

    def add_message(self, session_id, role, content, stage=None, score=None):
        with SessionLocal() as db:
            msg = Message(
                session_id=session_id, role=role, content=content,
                stage=stage, score=score,
            )
            db.add(msg)
            db.commit()

    def update_title(self, session_id, title):
        with SessionLocal() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.title = title
                db.commit()
```

**标题重写规则**：默认"新对话"，首条用户消息后截断改标题（最多18字符），用户已手改则不再覆盖。

### 9.4 会话历史接入

**核心概念**：RAG和Web搜索支持多轮追问，将历史消息拼入Prompt。

```python
class RAGService:
    def generate(self, question, rows, history=None):
        context = ''
        for row in rows:
            context += f"\n===== 《{row.doc_file_name}》第{row.doc_index}段 ======\n{row.content}"

        messages = [{"role": "system", "content": RAG_PROMPT.format(context=context, question=question)}]

        # 加入历史对话
        if history:
            for h in history[-config.HISTORY_LIMIT:]:
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": question})
        return self.llm.chat(messages)
```

**配置参数**：
```env
HISTORY_LIMIT=6
```

### 9.5 查询改写和HyDE

**核心概念**：
- **查询改写**：用LLM将用户追问改写为独立完整的问题，再检索
- **HyDE（Hypothetical Document Embeddings）**：先让LLM生成假设性答案文档，用该文档做Embedding检索

**查询改写**：
```python
REWRITE_PROMPT = """根据对话历史，将用户的追问改写为一个独立、完整的问题。
只输出改写后的问题，不要其他解释。

对话历史：
{history}

用户追问：{question}

改写后的问题："""

class QueryRewriter:
    def rewrite(self, question, history):
        prompt = REWRITE_PROMPT.format(history=history, question=question)
        return self.llm.chat([{"role": "user", "content": prompt}])
```

**HyDE**：
```python
HYDE_PROMPT = """请根据以下问题，写一段可能包含答案的文本（不需要正确，只需要语义相关）。

问题：{question}

假设性答案："""

class HyDEService:
    def generate_hypothetical(self, question):
        prompt = HYDE_PROMPT.format(question=question)
        return self.llm.chat([{"role": "user", "content": prompt}])

    def retrieve(self, question):
        hyp = self.generate_hypothetical(question)
        vector = self.embedding.embed([hyp])[0]
        # 用假设文档的向量做检索
        results = self.milvus.client.search(...)
        return results
```

**策略选择**：默认走查询改写，HyDE作为备选策略（延迟增加、改写跑偏风险）。

---

## 十、评估与进阶

### 10.1 RAGAS评估

**核心概念**：RAGAS（RAG Assessment）评估框架，量化RAG系统质量。

**评估维度**：

| 维度 | 说明 | 衡量什么 |
|------|------|----------|
| 忠实度（Faithfulness） | 答案是否忠于上下文 | 有没有编造 |
| 答案相关性（Answer Relevancy） | 答案是否回答了问题 | 有没有跑题 |
| 上下文相关性（Context Relevancy） | 检索上下文是否相关 | 检索准不准 |
| 上下文召回率（Context Recall） | 相关信息是否被检索到 | 检索全不全 |

**评估流程**：
1. 准备评估集（问题、标准答案、参考上下文）
2. 用RAG系统跑评估集，收集输出
3. 用RAGAS计算各项指标
4. 对比不同配置的指标，指导优化

### 10.2 RAG优化策略

**优化方向**：

1. **分流策略**：
   - 闲聊 → Greeting短路
   - 知识库问题 → RAG检索
   - 时效性问题 → 联网搜索
   - 通过分类器或LLM路由

2. **切分优化**：
   - 调整CHUNK_SIZE / CHILD_CHUNK_SIZE / CHUNK_OVERLAP
   - CSV一行一父块；Markdown按段落优先；PDF先OCR再切分

3. **检索优化**：
   - 混合检索（稠密+稀疏+RRF）
   - 调整RAG_SCORE_THRESHOLD
   - 重排选片

4. **生成优化**：
   - Prompt工程：依据上下文、不足则说明
   - 限制HISTORY_LIMIT控长度与跑题
   - 上下文太多时降K、提阈值、压缩摘要

5. **bad case排查路径**：
   - 切分/入库 → 检索分数与片段 → 阈值与K → Prompt/历史 → 模型

---

## 十一、项目拓展

### 11.1 后台登录

**核心概念**：管理员后台，渲染登录页 → 登录验证 → 注销。

**数据模型**：
```python
class AdminUser(Base):
    __tablename__ = "admin_user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
```

**登录验证**：
```python
# services/auth.py
import hashlib, secrets

def hash_password(password):
    salt = secrets.token_hex(8)
    return f"{salt}${hashlib.sha256((salt + password).encode()).hexdigest()}"

def verify_password(password, password_hash):
    salt, hashed = password_hash.split("$")
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed

def require_login(request):
    token = request.cookies.get("session_token")
    if not token or token not in SESSIONS:
        return None
    return SESSIONS[token]
```

**路由**：
```python
@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = verify_user(username, password)
    if not user:
        return RedirectResponse("/login?error=1", status_code=302)
    token = secrets.token_hex(16)
    SESSIONS[token] = username
    response = RedirectResponse("/admin", status_code=302)
    response.set_cookie("session_token", token)
    return response

@router.get("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")
    if token and token in SESSIONS:
        del SESSIONS[token]
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session_token")
    return response
```

### 11.2 知识库管理

**核心概念**：后台上传文件 → 切分入库 → 列表展示 → 删除。

**VectorStore列表与删除**：
```python
# services/vector_store.py
class VectorStore:
    def list_files(self):
        rows = self.client.query(
            collection_name=self.collection_name,
            filter='file != ""',
            output_fields=["file"],
            limit=16384,
        )
        from collections import Counter
        counter = Counter((r.get("file") or "") for r in rows)
        return [{"file": name, "chunks": count} for name, count in sorted(counter.items()) if name]

    def delete_by_file(self, file_name):
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'file == "{file_name}"',
        )
```

**上传接口**：
```python
@router.post("/api/admin/kb/upload")
async def kb_upload(request: Request, file: UploadFile = File(...)):
    username = require_login(request)
    if not username:
        return RedirectResponse("/login", status_code=302)
    # 保存文件
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    dest = UPLOAD_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # 切分入库
    docs = load_and_chunk(dest)
    for d in docs:
        d.metadata["file"] = safe_name
    n = VectorStore().add_documents(docs)
```

### 11.3 线上质检打标

**核心概念**：后台会话回放 + 人工打标（good/ok/bad）。

**Message追加打标字段**：
```python
class Message(Base):
    # ...原有字段...
    label = Column(String(16), nullable=True)      # good / ok / bad
    note = Column(String(512), nullable=True)       # 备注
    labeled_by = Column(String(64), nullable=True)  # 标注人
    labeled_at = Column(DateTime, nullable=True)     # 标注时间
```

**后台列表与打标**：
```python
class ConversationService:
    def list_sessions_admin(self, label_filter=None):
        """label_filter: None | unlabeled | good | ok | bad"""
        with SessionLocal() as db:
            rows = db.query(ChatSession).order_by(desc(ChatSession.updated_at)).all()
            items = []
            for s in rows:
                msgs = db.query(Message).filter(
                    Message.session_id == s.id, Message.role == "assistant"
                ).all()
                unlabeled_count = len(msgs) - len([m for m in msgs if m.label])
                # 按筛选条件过滤...
                items.append({...})
            return items

    def set_message_label(self, message_id, label, note, labeled_by):
        with SessionLocal() as db:
            row = db.query(Message).filter(Message.id == message_id).first()
            if label:
                row.label = label
                row.note = note
                row.labeled_by = labeled_by
                row.labeled_at = datetime.now()
            else:
                row.label = None  # 清空
            db.commit()
```

### 11.4 项目部署

**部署步骤**：
1. 启动Docker中间件（MySQL、Milvus及依赖组件）
2. 创建Python虚拟环境，安装依赖
3. 配置.env（MySQL/Milvus连接、模型参数、阈值、SerpAPI Key等）
4. 初始化与数据入库：
```bash
alembic upgrade head        # MySQL建表
python cli.py init-milvus   # Milvus建集合
python cli.py ingest-faq    # FAQ入库
python cli.py ingest-rag --file_name "学科知识问答.csv"
python cli.py ingest-rag --file_name "人工智能就业课课程大纲.md"
```
5. 启动服务：
```bash
python app.py
# 浏览器访问 http://127.0.0.1:8089
```

**冒烟验证**：
```bash
python cli.py test-clients
python cli.py test-ask --question="你好"
python cli.py test-ask --question="入门试学班学费多少钱？"
python cli.py test-ask --question="人工智能就业课有哪些课程优势"
```

### 11.5 面试考点梳理

**终态流水线**：
```
Greeting → TF-IDF FAQ → RAG(阈值+历史) → 联网 → Fallback
（工程化：流式 / sessions+messages / 多会话）
```

**面试要点**：
- 先结论 → 本项目怎么做 → 取舍与改进
- 能画流水线：阶段名 + 阈值条件
- 带数字：阈值、K、chunk、HISTORY_LIMIT、评估对比
- 不会的点说边界，转成「上线如何验证」

**高频考点**：

| 类别 | 考点 |
|------|------|
| 项目总览 | 教育场景智能助教，FAQ走TF-IDF，知识库走向量检索 |
| RAG原理 | 解决私有知识与时效，切分质量、检索噪声、阈值调参 |
| 检索策略 | BM25改进点、COSINE相似度、混合检索RRF |
| 生成与Prompt | 依据上下文、不足则说明、限制HISTORY_LIMIT |
| 工程化 | sessions+messages分离、REST接口、标题重写规则 |
| 评估优化 | RAGAS四维度、先基线再优化、分类路由 |

**高频追问**：
- 为什么选Milvus：面向向量检索、易扩容
- 换Embedding要全量重入
- 权限隔离：元数据带tenant/user，检索加filter
- 高并发：限流排队、热点缓存、超载降级
- Prompt注入防护：系统指令与用户内容分离

---

## 十二、前后端拓展

> 本章对应讲义**第十一章（11.1 ~ 11.6）**，是 RAG 主体之外的**前端与服务端渲染**补充线：HTML → CSS → 静态页实战 → Tailwind → FastAPI 模板 → 综合项目。
> 目的：能独立看懂并改动 `edu_rag` 的前端页面（`templates/`），并能独立交付一个小 Web 项目。

### 12.1 HTML 基础（11.1）

**定位**：HTML 用标签描述页面**结构与内容**，不负责漂亮（漂亮交给 CSS）。

#### 最小合法页面

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <title>我的第一页</title>
</head>
<body>
    <h1>你好，HTML</h1>
    <p>这是正文段落。</p>
</body>
</html>
```

| 部分 | 作用 |
|------|------|
| `<!DOCTYPE html>` | 声明按 HTML5 解析 |
| `<html lang>` | 整页根节点 |
| `<head>` | 元信息：编码、标题、外链 CSS/JS（不直接显示） |
| `<meta charset="utf-8">` | 避免中文乱码 |
| `<title>` | 浏览器标签页标题 |
| `<body>` | 用户可见的内容 |

#### 结构与分区（语义化）

```html
<header>页头 / 站点名</header>
<main>
    <section><h2>教育经历</h2><p>…</p></section>
    <section><h2>工作经历</h2><div>一块容器，本身无语义</div></section>
</main>
<footer>页脚</footer>
```

- `h1`~`h6` 标题（一页通常一个 `h1`）、`p` 段落
- `div` 通用块容器；有明确含义时优先 `section` / `header` / `main` / `footer` / `nav` / `aside`

#### 文本、链接、图片

```html
<p>姓名：<strong>张三</strong>，爱好 <em>阅读</em>。</p>
<p>
    邮箱：<a href="mailto:zhangsan@example.com">zhangsan@example.com</a>
    · 电话：<a href="tel:13800000000">138-0000-0000</a>
    · 主页：<a href="https://example.com" target="_blank" rel="noopener">个人站</a>
</p>
<img src="avatar.jpg" alt="头像" width="120" />
<hr />
<p>第一行<br />第二行</p>
<blockquote>引用一句自我评价</blockquote>
<p>代码片段：<code>print("hi")</code></p>
```

| 标签 | 用途 |
|------|------|
| `strong` / `b` | 加粗（语义强调优先 `strong`） |
| `em` / `i` | 斜体 |
| `span` | 行内小包裹，常配合 CSS class |
| `a` | 超链接；`href` 支持网址 / `mailto:` / `tel:` / `#锚点` |
| `img` | 图片；**必写 `alt`** |
| `br` / `hr` | 换行 / 水平分割线 |

> `target="_blank"` 时同时写 `rel="noopener"`，防止新页面通过 `window.opener` 操作原页面。

#### 列表与表单

```html
<ol><li>2018–2022 某某大学 · 计算机</li></ol>   <!-- 有序：时间线、步骤 -->
<ul><li>Python</li><li>SQL</li></ul>            <!-- 无序：技能、荣誉 -->

<form action="/submit" method="post">
    <label>
        留言
        <textarea name="message" rows="3" placeholder="请输入内容"></textarea>
    </label>
    <button type="submit">提交</button>
</form>
<input type="text" name="username" placeholder="用户名" />
<input type="password" name="password" />
<input type="email" name="email" />
```

- `name`：提交时的字段名（后端靠它取值）；`type`：`text` / `password` / `email` / `submit`

### 12.2 CSS 基础（11.2）

**定位**：CSS（Cascading Style Sheets）负责**外观**：颜色、字号、间距、布局。顺序：引入 → 选择器 → 文字 → 盒模型 → 横向居中 → `display` → Flex。

#### 三种引入方式

| 方式 | 写法 | 适用 |
|------|------|------|
| 内联 | `<p style="color:red">` | 临时调试 |
| 页内 | `<head>` 里 `<style>…</style>` | 单页 demo |
| 外链 | `<link rel="stylesheet" href="style.css" />` | 多页站点、样式复用 |

#### 选择器

```css
p { font-size: 16px; }        /* 标签选择器 */
.card { background: #f5f5f5; } /* class：可复用，最常用 */
#profile { max-width: 720px; } /* id：一页通常唯一 */
.card p { … }                  /* 后代：卡片里的段落 */
```

#### 文字与盒模型

```css
h1 { font-size: 32px; font-weight: 700; color: #0f172a; text-align: center; }
p  { font-size: 15px; line-height: 1.7; color: #334155; }
a  { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }

* { box-sizing: border-box; }   /* width 含 padding/border */
.card {
    width: 480px; max-width: 100%;
    padding: 16px 20px;         /* 两值 = 上下 / 左右 */
    margin: 12px auto;          /* 左右 auto = 盒子横向居中 */
    border: 1px solid #e5e5e5; border-radius: 8px; background: #fff;
}
```

#### 横向居中三板斧（易混点）

| 居中对象 | 写法 |
|----------|------|
| 一行文字 | 父级 `text-align: center` |
| 定宽盒子 | `width`/`max-width` + `margin: 0 auto` |
| Flex 子项 | 父级 `display:flex; justify-content:center`（+ `align-items:center` 垂直居中） |

#### display 与 Flex

| `display` 值 | 表现 |
|--------------|------|
| `block` | 独占一行，可设宽高（`div`/`p`/`h1`） |
| `inline` | 跟在文字后，宽高不好控（`span`/`a`） |
| `inline-block` | 可横排，又可设宽高 |
| `flex` | 子项按弹性盒排布 |

```css
.row { display: flex; justify-content: flex-start; align-items: center; gap: 12px; }
.header { display: flex; justify-content: space-between; align-items: center; }
.side { width: 220px; flex-shrink: 0; }
.main { flex: 1; min-width: 0; }
```

| Flex 属性 | 记忆 |
|-----------|------|
| `justify-content` | 主轴（默认左右）：`flex-start` / `center` / `flex-end` / `space-between` |
| `align-items` | 交叉轴（上下）：`stretch` / `flex-start` / `center` / `flex-end` |
| `gap` | 子项间距 |
| `flex: 1` | 写在子项上，吃掉剩余宽度 |

> 左右布局口诀：**父 `flex` + 子一边定宽 + 另一边 `flex: 1`**

### 12.3 静态页综合实战（11.3）

用 HTML + CSS 做**个人介绍页** `profile.html`（完整骨架见配套笔记《HTML基础结构笔记》十三节）。

**自检清单**：

- 完整文档骨架；标题与正文大小有区分
- 含加粗、斜体、超链接；有序 + 无序列表都用到
- 至少两块「作业要求之外」的补充（简介 / 技能 / 作品链接）
- 浏览器打开版式正常，无重叠、无乱码

可选延伸：Tailwind 美化（11.4）、服务端渲染（11.5）、`@media` 响应式、少量 JS。

### 12.4 Tailwind CSS（11.4）

**工具类优先**：不写 `.card { padding: 16px }`，直接在 HTML 上拼 class。

| 传统 CSS | Tailwind |
|----------|----------|
| `padding: 16px` | `p-4`（1 单位 = 4px） |
| `font-size: 14px` | `text-sm` |
| `color: #64748b` | `text-slate-500` |
| `display:flex; gap:12px` | `flex gap-3` |
| `margin: 0 auto` | `mx-auto` |
| `align-items:center` | `items-center` |
| `justify-content: space-between` | `justify-between` |
| `flex: 1` / `flex-shrink: 0` | `flex-1` / `shrink-0` |
| `border-radius` | `rounded-lg` |

```html
<script src="https://cdn.tailwindcss.com"></script>
<script>
    tailwind.config = { theme: { extend: { colors: { brand: "#047857" } } } };
</script>

<div class="flex min-h-screen">
    <aside class="flex w-64 shrink-0 flex-col bg-slate-900 p-4 text-slate-200">侧栏</aside>
    <main class="flex-1 p-6">主区</main>
</div>
```

- class 变长时按「布局 → 间距 → 文字 → 颜色」顺序写
- CDN 适合学习与 demo；生产用「安装 + 构建」裁剪未用 class

### 12.5 FastAPI 基础（11.5）

**本节范围**：服务端渲染页面这条线（对应 `edu_rag` 的 `templates/` 渲染方式）。

- **uvicorn**：ASGI 服务器，负责跑 FastAPI 应用
- **Jinja2**：模板引擎，把 HTML 与变量拼成最终页面

```bash
pip install fastapi uvicorn jinja2
```

#### 基本结构

```python
# app.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def index():
    return {"msg": "Hello, FastAPI"}   # 返回 dict → JSON

@app.get("/hello")
def hello():
    return "纯文本也可以"

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
```

| 写法 | 含义 |
|------|------|
| `app = FastAPI()` | 创建应用实例 |
| `@app.get("/路径")` | 登记 GET 路由 |
| 返回 `dict` | 自动变 JSON 响应 |
| `reload=True` | 改代码自动重启（开发用） |

启动后：`http://127.0.0.1:8000`、文档页 `/docs`

#### 模板渲染

```python
# app.py
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "首页"},
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
```

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8" /><title>{{ title }}</title></head>
<body><h1>{{ title }}</h1></body>
</html>
```

- 路由函数需接收 `request: Request`；`name` 相对 `templates/`；`context` 传数据字典

#### 数据渲染（变量 / 条件 / 循环）

```python
@app.get("/users")
def users(request: Request):
    user_list = [{"name": "张三", "role": "学员"}, {"name": "李四", "role": "助教"}]
    return templates.TemplateResponse(
        request=request, name="users.html",
        context={"title": "用户列表", "users": user_list, "show_tip": True},
    )
```

```html
<h1>{{ title }}</h1>
{% if show_tip %}<p>以下为演示数据</p>{% endif %}
<ul>
    {% for u in users %}
    <li>{{ u.name }} - {{ u.role }}</li>
    {% else %}
    <li>暂无数据</li>
    {% endfor %}
</ul>
```

| 模板语法 | 作用 |
|----------|------|
| `{{ x }}` | 输出变量（自动转义 HTML） |
| `{% if %}…{% endif %}` | 条件 |
| `{% for %}…{% else %}…{% endfor %}` | 循环（空数据时走 else） |

路径参数：`@app.get("/hello/{name}")` → `context={"name": name}`

#### Layout（模板继承）

```
templates/
├── layouts/base.html      # 总布局
├── partials/nav.html      # 可 include 的片段
├── index.html
└── about.html
```

```html
<!-- layouts/base.html -->
<title>{% block title %}站点{% endblock %}</title>
{% block head %}{% endblock %}
{% include "partials/nav.html" %}
<div class="wrap">{% block content %}{% endblock %}</div>

<!-- index.html -->
{% extends "layouts/base.html" %}
{% block title %}首页{% endblock %}
{% block content %}<h1>{{ title }}</h1><p>{{ message }}</p>{% endblock %}
```

| 指令 | 作用 |
|------|------|
| `{% extends %}` | 继承基座 |
| `{% block %}…{% endblock %}` | 基座预留坑位，子页填坑 |
| `{% include %}` | 嵌入公共 HTML 片段 |

#### 静态文件与 `url_for`

```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
```

```html
<link rel="stylesheet" href="{{ url_for('static', path='style.css') }}" />
<img src="{{ url_for('static', path='logo.png') }}" alt="logo" />
<a href="{{ url_for('index') }}">首页</a>
```

> 模板里**不要写死** `/static/...` 与端口路径，统一用 `url_for` 反查。

### 12.6 综合项目：真博客（11.6）

**定位**：把 HTML / CSS / Tailwind / FastAPI / SQLAlchemy / Alembic 串成完整工程，交付**可持久化**的博客系统。课堂方式：**写清需求与验收 → AI 生成工程 → 跑通并改到达标**。

| 不做（假项目） | 要做（真博客） |
|----------------|----------------|
| 内存 `list` / 重启清空 | MySQL + SQLAlchemy + Alembic 持久化 |
| 只有列表 + 详情 + 一个表单 | 注册登录、发文改删、分类标签、评论、分页搜索 |
| 讲义全文粘贴即交卷 | 独立仓库 + `README` 一键跑起来 |

#### 产品范围

**必须**：用户注册登录注销（密码哈希）、文章 CRUD + 草稿/发布、分类与多标签筛选、评论（作者可删）、标题搜索、Jinja2 layout + Tailwind、`.env` + `requirements.txt` + `README` + Alembic 可 `upgrade head`。

**加分**：Markdown 渲染（注意 XSS 过滤）、封面上传、文章置顶 / 浏览量、后台管理、Typer CLI 种子数据。

#### 技术栈与目录

- FastAPI + uvicorn + Jinja2 + python-multipart
- SQLAlchemy + Alembic + MySQL（Docker 起库）
- Tailwind CDN（或自建 CSS）
- Session Cookie 登录（课堂推荐，比 JWT 直观）

```
blog/
├── .env / .env.example
├── requirements.txt
├── README.md
├── app.py / config.py / cli.py
├── alembic.ini + alembic/
├── db/            # base.py / session.py / models.py
├── services/      # 鉴权、文章、评论
├── routers/       # 按模块拆路由
├── templates/     # layouts/ partials/ …
└── static/
```

#### 数据模型（最低集）

| 表 | 关键字段 |
|----|----------|
| `users` | `id`、`username`(唯一)、`password_hash`、`created_at` |
| `categories` | `id`、`name`(唯一)、`slug` |
| `tags` | `id`、`name`(唯一) |
| `posts` | `id`、`title`、`slug`、`content`、`status`(draft/published)、`author_id`、`category_id`、`created_at`、`updated_at` |
| `post_tags` | 多对多：`post_id` + `tag_id` |
| `comments` | `id`、`post_id`、`user_id`、`body`、`created_at` |

建表流程同第 4 章：改 Model → `alembic revision --autogenerate` → `alembic upgrade head`。

#### 页面与路由清单

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 已发布文章列表（分页、`?q=`、分类/标签筛选） |
| GET | `/posts/{slug}` | 详情 + 评论列表 |
| GET/POST | `/auth/register`、`/auth/login` | 注册 / 登录 |
| POST | `/auth/logout` | 注销 |
| GET/POST | `/posts/new` | 发文（需登录） |
| GET/POST | `/posts/{id}/edit` | 编辑（仅作者） |
| POST | `/posts/{id}/delete` | 删除（仅作者） |
| POST | `/posts/{slug}/comments` | 发表评论（需登录） |
| POST | `/comments/{id}/delete` | 删评论（本人） |

> 写成功后统一 `RedirectResponse(..., 303)`，避免刷新重复提交。

#### 用 AI 生成：总提示词（可直接使用）

```
请在当前目录生成一个可运行的个人博客项目，要求如下：

技术栈（必须遵守）：
- FastAPI + uvicorn + Jinja2Templates + python-multipart
- SQLAlchemy 2.x + Alembic + MySQL（连接串读 .env 的 DATABASE_URL）
- Session Cookie 登录；密码用哈希（如 passlib/bcrypt 或等价安全方案）
- 前端：Jinja2 layout + Tailwind CDN；静态资源用 StaticFiles + url_for
- 不要用内存 list 存业务数据；不要把密码明文入库

功能（必须全部实现）：
- 用户注册 / 登录 / 注销
- 文章 CRUD；status=draft|published；仅作者可改删
- 分类、标签（多对多）、评论
- 首页分页；标题搜索；按分类/标签筛选
- README 写清：装依赖、配 .env、alembic upgrade、启动命令、默认种子账号（若有）

目录分层：config / db(models+session) / routers / services / templates / static / alembic
先给出目录树与 Model，再写路由与模板；每完成一块说明如何本地验证。
```

**建议分步**（减少一次生成翻车）：骨架 → 用户 → 文章 → 评论+搜索分页 → 打磨（权限、空状态、404、README、种子数据）。

**必须亲手盯住的点**：

- AI 用内存字典 / JSON 文件顶替数据库 → 打回重做
- 未登录也能 POST 发文、改别人的文章 → 补依赖注入 / 中间件校验
- 模板里写死 `/static/...` 或硬编码端口 → 改成 `url_for`
- `alembic` 的 `env.py` 没挂 Model metadata → migrate 空转
- 评论原样输出危险 HTML → 默认转义或安全渲染（XSS）

#### 落地与验收

```bash
cd blog
pip install -r requirements.txt
cp .env.example .env     # 改 DATABASE_URL
alembic upgrade head
python app.py            # http://127.0.0.1:8000
```

**验收清单**：

- 重启服务后，之前发布的文章 / 评论仍在
- 未登录访问「写文章」跳登录或 403；登录后可发草稿与已发布
- 作者可编辑 / 删除自己的文，他人不可
- 分类、标签筛选与标题搜索可用；列表有分页
- 评论可发、可删（本人）；详情页展示评论
- 新环境按 README 能从零跑通（含 migrate）

#### 与前面各节的对应

| 能力 | 来源 |
|------|------|
| HTML / 表单 / 语义结构 | 11.1 |
| 排版与布局直觉 | 11.2 |
| 静态页综合 | 11.3 |
| Tailwind 工具类 | 11.4 |
| FastAPI、Jinja2、layout、`url_for` | 11.5 |
| SQLAlchemy、Alembic、`.env`、分层 | 第 4 章 |
| 登录会话思路（可对照） | 10.1 |

### 12.7 配套作业：E1 前端 HTML 基础（10.6）

用纯 HTML 写一个**个人介绍页**（学完 11.2 后建议补一版基础样式）。

**内容要求**：

- 完整结构：`<!DOCTYPE html>`、`<html>`、`<head>`（含 `charset`、`title`）、`<body>`
- 个人信息区：姓名、头像、电话、邮箱（可用 `tel:` / `mailto:`）
- 分块展示：教育经历、工作经历、获得荣誉（`<h2>` / `<section>` / `<div>`）
- 再补充至少两项：个人简介 / 技能标签 / 兴趣爱好 / 作品链接 / 自我评价

**标签要求**：

- 标题与正文有大小区分（`h1`~`h3` 与 `p`）
- 至少用到加粗、斜体、超链接；任选 `img`、`br`、`hr`、`span`、`blockquote`、`code`
- 同时使用有序列表（`ol`）与无序列表（`ul`）

**提交**：单个 `.html` 文件（或压缩包），浏览器打开可正常显示；内容可用真实信息或合理虚构，勿泄露敏感隐私。

### 12.8 本章速查与小结

```
HTML：骨架 → 语义分区(header/main/section/footer) → 文本/链接/图片 → 列表 → 表单
CSS ：引入 → 选择器 → 文字/颜色 → 盒模型(padding/margin/border) → 居中 → Flex
居中：文字 text-align / 盒子 margin:0 auto / 子项 justify-content:center
Flex：父 display:flex + justify-content(主轴) + align-items(交叉轴) + gap；子 flex:1
Tailwind：p-4 text-sm bg-white flex items-center justify-between flex-1 mx-auto
FastAPI：FastAPI() + @app.get + uvicorn.run；Jinja2Templates + TemplateResponse(request, name, context)
Jinja2：{{ 变量 }} {% if %} {% for %} {% extends %} {% block %} {% include %} url_for
博客：持久化 + 账号权限 + CRUD + 可交接 README
```

**小结**：

1. HTML 管结构、CSS 管外观、JS 管行为；本章覆盖前两者
2. 居中与 Flex 是页面「改得动」的分水岭
3. FastAPI + Jinja2 就是 `edu_rag` 前端页面的底层套路：`templates/` + `url_for` + layout
4. 综合项目验收不看「粘了多少代码」，看主流程是否稳定可复现

---


## 附录

### A. 项目依赖

```
# edu_rag
fastapi>=0.110.0
uvicorn>=0.27.0
jinja2>=3.1.0
pymysql>=1.1.0
sqlalchemy>=2.0.0
alembic>=1.13.0
python-dotenv>=1.0.0
redis>=5.0.0
pymilvus>=2.4.0
openai>=1.40.0
numpy>=1.24.0
jieba>=0.42.1
scikit-learn>=1.3.0
pandas>=2.0.0
langchain-text-splitters>=0.2.0
unstructured[all-docs]>=0.11.0
langchain-unstructured>=0.1.0
google-search-results>=2.4.2
```

### B. 关键配置参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MYSQL_HOST | localhost | MySQL主机 |
| MYSQL_PORT | 3306 | MySQL端口 |
| MYSQL_USER | root | MySQL用户 |
| MYSQL_DATABASE | edu_rag_project | MySQL数据库名 |
| MILVUS_HOST | localhost | Milvus主机 |
| MILVUS_PORT | 19530 | Milvus端口 |
| MILVUS_COLLECTION_NAME | edu_rag | Milvus集合名 |
| EMBEDDING_DIM | 1024 | Embedding向量维度 |
| LLM_MODEL | qwen-plus | LLM模型名 |
| EMBEDDING_MODEL | text-embedding-v3 | Embedding模型名 |
| TFIDF_THRESHOLD | 0.5 | FAQ TF-IDF阈值 |
| CHUNK_SIZE | 1000 | 父块字符数 |
| CHUNK_OVERLAP | 100 | 父块重叠 |
| CHILD_CHUNK_SIZE | 200 | 子块字符数 |
| CHILD_CHUNK_OVERLAP | 20 | 子块重叠 |
| RETRIEVAL_K | 5 | 检索TopK |
| RAG_SCORE_THRESHOLD | 0.45 | RAG向量阈值 |
| ES_TOP_K | 10 | ES召回数 |
| RERANK_MODEL | qwen3.7-text-rerank | 重排模型 |
| RERANK_TOP_N | 5 | 重排返回数 |
| HISTORY_LIMIT | 6 | 历史消息数 |
| LOG_FILE | storage/logs/app.log | 日志文件路径 |

### C. CLI命令汇总

| 命令 | 说明 |
|------|------|
| `python cli.py init-milvus` | 创建Milvus集合 |
| `python cli.py ingest-faq` | 导入FAQ到MySQL |
| `python cli.py ingest-rag --file_name xxx` | RAG知识库入库 |
| `python cli.py test-clients` | 测试客户端连接 |
| `python cli.py test-ask --question="xxx"` | 测试问答流程 |
| `python cli.py test-es --question="xxx"` | 测试ES召回 |
| `python cli.py test-hybrid --question="xxx"` | 测试多路RRF |
| `python cli.py add-user --username xxx --password xxx` | 添加后台用户 |

### D. API接口汇总

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 |
| `/health` | GET | 健康检查 |
| `/api/ask` | POST | 非流式问答 |
| `/api/stream` | WS | 流式问答 |
| `/api/sessions` | GET/POST | 会话列表/新建 |
| `/api/sessions/{id}` | GET/PATCH/DELETE | 会话详情/改标题/删除 |
| `/api/sessions/{id}/messages` | GET | 消息列表 |
| `/login` | GET/POST | 登录 |
| `/logout` | GET | 注销 |
| `/admin` | GET | 后台首页 |
| `/admin/kb` | GET | 知识库管理 |
| `/api/admin/kb/upload` | POST | 上传知识库文件 |
| `/api/admin/kb/delete` | POST | 删除知识库文件 |
| `/admin/sessions` | GET | 会话质检列表 |
| `/admin/sessions/{id}` | GET | 会话回放详情 |
| `/api/admin/messages/{id}/label` | POST | 打标/清空标注 |
