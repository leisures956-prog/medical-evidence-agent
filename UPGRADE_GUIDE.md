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
