# Search Answer API

## 当前状态

当前版本：v0.2.5（轻量证据知识图谱阶段）

- 已实现：PubMed 富元数据检索、医学文档结构、Embedding/混合检索框架、LLM 接口与引用编号检查。
- 已实现：`/evidence` 摘要级证据接口、PMID/标题/摘要原文片段追踪和严格证据校验。
- 已实现：`/compare` 多文献比较接口，并标记校正文章、综述文章和摘要缺失。
- 已实现：证据级引用校验、`[n:evidence_id]` 精确引用格式和 12 条固定评测问题。
- 已实现：从结构化证据生成轻量 JSON 知识图谱，保留 PMID、来源位置和原文片段。
- 已验证：45 个本地自动化测试通过；真实 PubMed 检索成功，能够返回论文标题、摘要、作者和来源链接。
- 待验证：真实 Embedding 检索效果、真实 LLM 端到端回答。
- 待开发：LangGraph 流程编排和最终 GitHub 展示。

引用检查现在可以把每个引用绑定到具体 PMID、标题/摘要片段和 `evidence_id`；它仍然不替代医学事实真伪判断。

这是一个用于学习“检索增强问答（RAG）”的 FastAPI 项目。当前版本先完成稳定的基础层：

- `/health`：健康检查
- `/search`：本地 JSON 文档检索，或检索 PubMed
- `/search` 的 `source` 还支持 `vector`，用于本地 Embedding 向量检索
- `/search` 的 `source` 支持 `pubmed_hybrid`，先召回 PubMed 文献，再使用关键词 + 向量排序
- `/answer`：根据检索结果返回带来源编号的回答
- `/evidence`：PubMed 混合召回后，返回只基于标题/摘要的结构化证据 JSON
- `/compare`：比较多篇 PubMed 文献，并返回结构化证据和文章质量标记
- 本地文档从 `data/documents.json` 读取，不再写死在 Python 文件里
- PubMed 请求带超时、HTTP 异常处理和自动化测试

## 使用 Python 3.11 环境

你当前已经创建好的环境在项目的 `.conda/search_api311` 目录中。若已经设置了 PowerShell 快捷函数，可以运行：

```powershell
api311
```

也可以直接激活：

```powershell
conda activate ".\.conda\search_api311"
```

然后安装依赖并运行测试：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

启动开发服务器：

```powershell
python -m uvicorn app.main:app --reload
```

浏览器打开 <http://127.0.0.1:8000/docs>。

## 搜索示例

本地文档搜索：

```json
{
  "query": "Python 列表",
  "limit": 5,
  "source": "local"
}
```

PubMed 搜索：

```json
{
  "query": "lung cancer immunotherapy",
  "limit": 3,
  "source": "pubmed"
}
```

PubMed 混合排序：

```json
{
  "query": "lung cancer immunotherapy",
  "limit": 5,
  "candidate_limit": 30,
  "source": "pubmed_hybrid"
}
```

`hybrid` 继续只表示本地文档混合检索；`pubmed_hybrid` 才表示 PubMed 候选文献的混合排序。

本地向量检索需要先安装可选依赖：

```powershell
python -m pip install -r requirements-rag.txt
```

第一次使用 `source: "vector"` 时，会自动下载并缓存多语言 Embedding 模型
`paraphrase-multilingual-MiniLM-L12-v2`。在 Swagger 的 `/search` 中使用：

```json
{
  "query": "有序、可变的数据结构",
  "limit": 3,
  "source": "vector"
}
```

如果尚未安装可选依赖，接口会返回 `503`，而不是让整个服务崩溃。

## 摘要级证据

`/evidence` 固定使用 `pubmed_hybrid` 检索。默认 `metadata_only` 模式不调用大模型，
用于先检查 PMID、标题证据和“摘要未提及”字段：

```json
{
  "question": "lung cancer immunotherapy",
  "top_k": 1,
  "candidate_limit": 20,
  "mode": "metadata_only"
}
```

如果已经配置 `LLM_API_KEY` 和 `LLM_MODEL`，可以使用 `strict_llm`。模型只能看到标题和摘要，
每个非空字段必须绑定 `evidence_ids`，每个 `exact_text` 必须逐字出现在标题或摘要中；
否则接口会拒绝返回，避免把未验证的推断当成证据。

```json
{
  "question": "lung cancer immunotherapy",
  "top_k": 1,
  "candidate_limit": 20,
  "mode": "strict_llm"
}
```

每条记录包含 `pmid`、`source_url`、`title`、`research_object`、`disease`、`method`、
`dataset`、`main_findings`、`limitations` 和 `evidence_spans`。摘要未提及的字段返回
`value: null`，并标记 `not_reported_in_abstract`；没有摘要时标记
`abstract_unavailable`。当前校验保证可追溯性，不替代医学事实核查。

## 文献比较

`/compare` 默认比较 2～5 篇 PubMed 文献：

```json
{
  "question": "lung cancer immunotherapy",
  "top_k": 3,
  "candidate_limit": 30,
  "mode": "metadata_only"
}
```

返回的 `comparison_table` 每行包含 PMID、标题、混合检索分数、结构化证据和 `markers`：

- `correction`：标题包含 Correction、Erratum、Retraction 等校正/撤稿提示；
- `review`：标题包含 Review、Systematic Review、Meta-analysis 等综述提示；
- `abstract_missing`：PubMed 没有摘要。

这些标记是透明的标题启发式和摘要是否存在检查，不会改变原有 `pubmed_hybrid` 分数，
也不会改变本地 `hybrid` 的含义。摘要缺失或未提及的比较字段仍然返回 `null`。

## 引用验证

基础引用格式是 `[1]`。当一个来源只有一个证据片段时，系统可以自动绑定该片段；
当一个来源有多个片段时，必须使用精确格式：

```text
研究结果得到改善。[1:finding-1]
```

引用绑定结果会保留：

- `citation_number`
- `pmid`
- `evidence_id`
- `source_section`
- `exact_text`
- `source_url`

`citation_check_node` 和 `GroundedAnswerer(..., evidence_records=...)` 都会拒绝越界引用、
不存在的 `evidence_id`，以及无法唯一定位证据片段的 `[n]`。

## 固定评测

固定问题保存在 `data/evaluation_questions.json`，当前包含 12 个医学检索问题。
在已经配置 PubMed 和 Embedding 环境的 PowerShell 中运行：

```powershell
python -m app.evaluation --mode metadata_only --top-k 3 --candidate-limit 20 --output evaluation_results.json
```

评测会记录：召回成功率、主题词匹配率、JSON 完整率、引用有效率、摘要证据覆盖率、
平均响应时间、模型调用次数和估计成本。`metadata_only` 基线不会调用大模型，因此
`llm_calls` 和成本为 0，摘要证据覆盖率为 0 是预期基线，不代表程序报错。

## 轻量证据知识图谱

`app/knowledge_graph.py` 只接受已经通过结构校验的 `StructuredEvidence`，不会从未验证的
自由文本猜测实体。当前包含四类节点：

- `Paper`：以 PMID 为唯一标识，并保留 PubMed 链接；
- `Method`：摘要中明确报告的方法；
- `Disease`：摘要中明确报告的疾病；
- `Dataset`：摘要中明确报告的数据集。

论文与方法、疾病、数据集之间的每条关系都保留 `pmid`、`evidence_id`、
`source_section`、`evidence_text` 和 `source_url`。摘要未提及的字段不会创建节点或关系，
因此图谱不会把 `null` 当作事实。当前使用纯 JSON，不需要安装 NetworkX 或 Neo4j：

```python
from app.knowledge_graph import build_evidence_graph, write_graph_json

# records 应来自 /evidence 的结构化结果，或 EvidenceExtractor.extract() 的结果列表
graph = build_evidence_graph(records)
write_graph_json(graph, "artifacts/evidence_graph.json")
print(graph.paper_count, graph.relation_count)
```

输出文件可以直接作为 GitHub 展示样例或后续可视化的输入。它是证据图谱，不是完整医学知识图谱；
当前不做全文推断、实体链接、Neo4j 部署或 GraphRAG。

PubMed 连接使用 NCBI E-utilities 的 ESearch 和 ESummary。NCBI 建议在程序化访问时提供联系邮箱；可以在 PowerShell 中设置：

```powershell
$env:NCBI_EMAIL = "your-email@example.com"
# 如果有 NCBI API key，再设置：
# $env:NCBI_API_KEY = "your-api-key"
```

## Git 初始化

```powershell
git init
git add .
git commit -m "chore: initialize search API project"
```

## 后续 RAG 路线

建议按以下顺序推进，不要在检索层还不稳定时直接加入 Agent：

1. 当前阶段：本地数据源、PubMed 适配器、异常处理和测试。
2. Embedding 阶段：给文档切块，为每个块保存向量和来源元数据；先用本地 `sentence-transformers`，再接 FAISS 或 Chroma。
3. RAG 阶段：向量召回 + 关键词召回，合并、去重、重排，并把 `id/title/url` 一起传给回答器。
4. 大模型阶段：增加一个可替换的 LLM provider；提示词要求只根据检索上下文回答，并用 `[1]`、`[2]` 对应来源。
5. LangGraph 阶段：把检索、重排、回答、引用检查做成节点，再添加工具调用和重试分支。

Embedding 和大模型需要先确定是使用本地模型，还是 OpenAI 兼容接口；因此不在基础版本里强行下载大型模型或要求 API key。
# 升级说明

下一阶段模块已经接入原项目。网页请求示例与配置说明见 [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md)。

