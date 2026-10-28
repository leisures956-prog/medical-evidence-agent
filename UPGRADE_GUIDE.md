# 下一阶段已接入

启动目录仍然是 D:\Codex\medical-evidence-agent\python_proj\search_api。
升级包原目录保留作为参考，不需要在那里启动。

## 已有环境启动

激活 api311 后运行：

```powershell
cd D:\Codex\medical-evidence-agent\python_proj\search_api
python -m pytest -q
python -m uvicorn app.main:app --reload
```

打开 http://127.0.0.1:8000/docs。

## 网页操作

展开 POST /search，点击 Try it out，修改 JSON，再点击 Execute。

PubMed（不需要 Embedding 模型）：
```json
{"query":"lung cancer immunotherapy","limit":3,"source":"pubmed"}
```
检查 content 摘要、authors、year、doi、journal 和 url。部分文章没有摘要或 DOI，这是允许的。

PubMed 混合排序（需要本地 Embedding 依赖和模型）：
```json
{"query":"lung cancer immunotherapy","limit":5,"candidate_limit":30,"source":"pubmed_hybrid"}
```
它先从 PubMed 召回候选文献，再使用现有的关键词 + 向量排序。`hybrid` 仍然只表示本地文档混合检索。

本地混合检索（需要本地 Embedding 依赖和模型）：
```json
{"query":"Python lists","limit":3,"source":"hybrid"}
```
local 是关键词检索，vector 是原有向量检索，hybrid 是本地关键词和向量融合，all 保持原有聚合含义。

展开 POST /answer，点击 Try it out。
不调用大模型的默认模式：
```json
{"question":"Python lists","top_k":3,"source":"local","mode":"extractive"}
```
配置大模型后才使用：
```json
{"question":"What evidence exists for lung cancer immunotherapy?","top_k":3,"source":"pubmed","mode":"llm"}
```
llm 模式返回 citations 和 sources。confidence=0 表示未估计置信度，不是医学结论概率。
引用检查只验证编号有效，不验证每个论断是否被文献支持，需要人工核对。

## 配置与数据

原 data/documents.json 保留；新 data/medical_documents.sample.json 只是结构示例，不自动加载。
大模型使用 LLM_API_KEY、LLM_MODEL、LLM_BASE_URL 环境变量。
.env.example 供参考；当前代码不自动读取 .env，可用 uvicorn --env-file .env（需相应依赖）。
模型首次使用可能下载，必须先设置 D 盘缓存并另行确认下载：
```powershell
$env:HF_HOME = 'D:\ml-cache\huggingface'
$env:HF_HUB_CACHE = 'D:\ml-cache\huggingface\hub'
$env:PIP_CACHE_DIR = 'D:\ml-cache\pip'
```
requirements-next-stage.txt 是补充依赖，不替代 requirements.txt。
本次升级不包含 LangGraph，也不自动安装依赖或下载模型。

## 阶段 0：测试与基线规则

项目测试已分为离线单元测试和显式联网集成测试：

```powershell
python -m pytest -m unit -q
python -m pytest -m integration -q
```

`tests/integration/` 中的 PubMed 测试默认跳过；确认网络可用后设置
`$env:RUN_PUBMED_INTEGRATION = "1"` 再运行。pytest 临时目录使用项目内的 `.pytest_tmp/`，
避免受系统 C 盘临时目录权限影响。`evaluation_results.json` 保留为本地原始评测输出，
`tests/test1.ipynb` 已移到 `examples/demo_knowledge_graph.ipynb` 作为演示材料。

## 阶段 2：摘要级证据接口

展开 POST `/evidence`，点击 Try it out，先用不需要大模型的安全模式验证流程：

```json
{
  "question": "lung cancer immunotherapy",
  "top_k": 1,
  "candidate_limit": 20,
  "mode": "metadata_only"
}
```

接口流程是：PubMed 召回 → `pubmed_hybrid` 混合排序 → 证据结构化。返回结果中的每个
`evidence_span` 都保留 PMID、`source_section`（`title` 或 `abstract`）、原文
`exact_text` 和来源链接。摘要没有明确提到的研究对象、方法、数据集、主要发现或局限性，
不会由程序补造，而是返回 `value: null`。

配置大模型后，可把 `mode` 改为 `strict_llm`。该模式会拒绝非法 JSON、PMID 不匹配、
引用不存在片段或 `exact_text` 不在标题/摘要中的结果。当前“严格”指来源可追溯，
不代表自动完成医学事实真伪判定。

## 阶段 4：文献比较

展开 POST `/compare`，可比较 2～5 篇 PubMed 文献：

```json
{
  "question": "lung cancer immunotherapy",
  "top_k": 3,
  "candidate_limit": 30,
  "mode": "metadata_only"
}
```

返回的 `comparison_table` 保留每篇文献的 PMID、标题、检索分数、结构化证据和文章标记。
其中 `correction`、`review` 和 `abstract_missing` 分别用于提示校正/撤稿相关标题、
综述标题和没有摘要的文献。它们只是质量提示，不会改变检索排序。

## 阶段 5～7：引用验证与固定评测

引用验证支持两种形式：

```text
[1]
[1:finding-1]
```

当一个来源对应多个证据片段时，必须指定具体的 `evidence_id`。校验结果会保留 PMID、
标题/摘要来源、原文片段和来源链接。它只检查可追溯性，不自动判断医学结论是否真实。

固定评测问题在 `data/evaluation_questions.json` 中，共 12 条。运行：

```powershell
python -m app.evaluation --mode metadata_only --top-k 3 --candidate-limit 20 --output evaluation_results.json
```

结果包括检索结果数量、主题词匹配、JSON 完整性、引用有效率、摘要证据覆盖率、响应时间
和调用成本。当前不调用大模型，成本基线为 0。

## 阶段 8：轻量证据知识图谱

`app/knowledge_graph.py` 提供 `build_evidence_graph()` 和 `write_graph_json()`。输入必须是
`StructuredEvidence` 列表，推荐直接使用 `/evidence` 返回的记录转换后的 Pydantic 对象。
图谱只生成 `Paper`、`Method`、`Disease` 和 `Dataset` 四类节点；每条关系携带 PMID、
`evidence_id`、`source_section`、原文 `evidence_text` 和来源链接。

示例：

```python
from app.knowledge_graph import build_evidence_graph, write_graph_json

graph = build_evidence_graph(records)
write_graph_json(graph, "artifacts/evidence_graph.json")
```

`null` 或摘要未报告字段不会生成关系。实现使用 JSON 和 Pydantic，不新增第三方依赖，
也不部署 Neo4j；后续可在此基础上增加可视化或图检索。

## 阶段 9：GitHub 展示材料

README 已整理项目架构 Mermaid 图、启动与测试命令、`/evidence` 和 `/compare` 的 JSON 示例、
固定评测摘要、知识图谱 JSON/PNG 示例和项目局限性。示例文件位于 `examples/`：

- `evidence_response.json`：摘要级证据接口返回示例；
- `compare_response.json`：文献比较及校正/综述/摘要缺失标记示例；
- `evaluation_summary.json`：12 条固定问题的基线评测摘要。

阶段 9 不改变检索、证据提取、引用校验或知识图谱逻辑。GitHub 演示时建议依次打开 `/health`、
`/search`、`/evidence`、`/compare`，再展示 `artifacts/evidence_graph_visualization.png`。
