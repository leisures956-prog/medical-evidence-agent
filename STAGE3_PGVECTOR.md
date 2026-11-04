# 阶段 3：持久化向量检索

本阶段在不改变现有 `vector`、`hybrid` 和 `pubmed_hybrid` 含义的前提下，增加：

- `vector_pg`：PostgreSQL + pgvector 语义检索；
- `hybrid_pg`：关键词召回与 pgvector 召回融合；
- 文献、摘要块、Embedding、PMID、版本和内容哈希的持久化；
- 可重复执行的增量 upsert、单语料版本删除和 HNSW 余弦索引。

## 1. 启动数据库

需要安装 Docker Desktop，然后在项目根目录运行：

```powershell
docker compose -f compose.yaml -f compose.vector.yaml up -d db
```

默认连接地址为：

```powershell
$env:VECTOR_DATABASE_URL = "postgresql://search:search@localhost:5433/search"
```

## 2. 安装可选依赖

在 `api311` 环境中执行：

```powershell
python -m pip install -r requirements-rag.txt
python -m pip install -r requirements-vector.txt
```

`requirements-vector.txt` 只包含 PostgreSQL 驱动，不会改变基础 FastAPI 环境。

## 3. 为冻结 PubMed 快照建立索引

Embedding 模型可以使用本地路径：

```powershell
$env:EMBEDDING_MODEL = "D:\ml-cache\models\paraphrase-multilingual-MiniLM-L12-v2"
python scripts/index_pubmed_snapshot_pg.py `
  --snapshot data/benchmark/pubmed_snapshot_v1 `
  --database-url $env:VECTOR_DATABASE_URL
```

脚本会把每篇文献的标题和摘要按固定参数切块，保存：

```text
PMID、标题、摘要、作者、年份、DOI、来源 URL
chunk_index、chunk_text、content_sha256
embedding_model、embedding_dimension、corpus_version
```

## 4. 使用 API

启动 API 后，在 `/docs` 的 `/search` 中选择新模式：

```json
{
  "query": "lung cancer immunotherapy",
  "limit": 5,
  "source": "vector_pg"
}
```

或：

```json
{
  "query": "lung cancer immunotherapy",
  "limit": 5,
  "source": "hybrid_pg"
}
```

如果没有配置连接地址、Embedding 模型或数据库，接口返回 `503`，不会影响本地
`local`、`vector`、`hybrid` 模式。

## 5. 测试

离线测试不需要数据库：

```powershell
python -m pytest -m unit tests/test_pgvector.py -q
```

有数据库和 `psycopg` 时，再运行集成测试：

```powershell
$env:PGVECTOR_DATABASE_URL = $env:VECTOR_DATABASE_URL
python -m pytest -m integration tests/integration/test_pgvector_integration.py -q
```

集成测试使用临时语料版本，结束后会删除测试数据。向量库的正式评测仍应使用同一
冻结语料和同一标签版本，并记录模型、切块参数、语料版本和延迟。
