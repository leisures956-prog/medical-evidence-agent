# Search API 下一阶段文件包

这是一个独立文件包，生成位置在：

```text
D:\Codex\search_api_next_stage_files
```

它没有修改原项目目录。请先阅读代码，再按需要把文件复制到你的项目中。

## 文件对应关系

```text
app/medical_models.py       医学文档、检索结果、回答数据结构
app/pubmed_client_v2.py     PubMed 摘要、作者、年份、DOI 解析
app/semantic_retriever.py   本地 Embedding 语义检索
app/hybrid_retriever.py     关键词 + 向量混合检索
app/grounded_llm.py         严格依据检索资料的 LLM 回答器
app/agent_nodes.py          后续可接 LangGraph 的普通 Python 节点
data/medical_documents.sample.json
                            医学文档元数据示例
requirements-next-stage.txt 下一阶段依赖
.env.example                环境变量示例
tests/test_next_stage.py    独立测试示例
```

## 推荐接入顺序

### 1. PubMed 富元数据

```python
from app.pubmed_client_v2 import PubMedClientV2

client = PubMedClientV2()
documents = client.search("lung cancer immunotherapy", limit=3)

for document in documents:
    print(document.title)
    print(document.abstract)
    print(document.authors)
    print(document.year, document.doi)
```

`PubMedClientV2` 使用 ESearch 找 PMID，再用 EFetch XML 获取摘要、作者、年份、期刊、关键词和 DOI。

### 2. 本地 Embedding 语义检索

在 `api311` 环境中安装：

```powershell
python -m pip install -r requirements-next-stage.txt
```

使用 `SemanticRetriever`：

```python
from app.medical_models import MedicalDocument
from app.semantic_retriever import SemanticRetriever

documents = [
    MedicalDocument(
        id="doc-1",
        source="local",
        title="示例医学文档",
        abstract="这里放摘要。",
    )
]

retriever = SemanticRetriever(documents)
hits = retriever.search("与摘要语义相近的问题", top_k=3)
```

模型默认是：

```text
paraphrase-multilingual-MiniLM-L12-v2
```

首次使用时会下载模型。建议先设置：

```powershell
$env:HF_HOME = "D:\ml-cache\huggingface"
$env:HF_HUB_CACHE = "D:\ml-cache\huggingface\hub"
```

### 3. 混合检索

```python
from app.hybrid_retriever import HybridRetriever

hybrid = HybridRetriever(
    documents=documents,
    semantic_retriever=retriever,
    semantic_weight=0.7,
)
hits = hybrid.search("问题", top_k=5)
```

混合分数为：

```text
最终分数 = 0.7 × 语义分数 + 0.3 × 关键词分数
```

### 4. 大模型回答

先设置 `.env` 中的变量；不要把真实 API key 提交到 Git：

```text
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=你的密钥
LLM_MODEL=你的模型名
```

然后：

```python
from app.grounded_llm import GroundedAnswerer, OpenAICompatibleProvider

answerer = GroundedAnswerer(OpenAICompatibleProvider())
result = answerer.answer("问题", hits)
print(result.answer)
print(result.citations)
```

提示词要求模型只能使用检索上下文，并在结论后写 `[1]`、`[2]`。如果回答没有有效引用，代码会报错，而不是静默返回无来源答案。

### 5. Agent 节点

`agent_nodes.py` 先只提供普通 Python 函数，不引入 LangGraph：

```python
from app.agent_nodes import answer_node, citation_check_node, retrieve_node

state = {"question": "问题"}
state = retrieve_node(state, hybrid, top_k=5)
state = answer_node(state, answerer)
state = citation_check_node(state)
```

等检索、回答和引用检查分别通过测试后，再把这三个函数接入 LangGraph 的 StateGraph。

## 注意

- `medical_documents.sample.json` 是结构示例，不是临床建议。
- PubMed 摘要、作者和 DOI 应以接口返回为准，不要手工编造。
- 目前向量索引在内存中；下一步可以加入持久化索引文件或向量数据库。
- 本文件包不包含 LangGraph，符合“最后再接入 Agent”的顺序。
