# 阶段 2：冻结语料和人工标签

这一目录保存后续检索实验共用的固定候选库。先冻结文献，再编辑标签；不要在实验过程中直接修改 `documents.jsonl`。

## 1. 生成候选快照

脚本必须明确选择离线输入或联网模式，避免测试时意外访问 PubMed。

```powershell
# 推荐先用离线候选 JSON 验证流程
python scripts/freeze_pubmed_corpus.py `
  --questions data/benchmark/questions_v1.json `
  --records path/to/pubmed_candidates.json `
  --output data/benchmark/pubmed_snapshot_v1 `
  --candidate-limit 20

# 确认网络、环境变量和 NCBI 配置后，才显式使用 --live
python scripts/freeze_pubmed_corpus.py `
  --questions data/benchmark/questions_v1.json `
  --live `
  --output data/benchmark/pubmed_snapshot_v1 `
  --candidate-limit 20
```

输出包括：

- `manifest.json`：语料版本、获取时间、哈希规则、候选池范围；
- `questions.json`：冻结的问题集；
- `documents.jsonl`：去重后的标题、摘要、PMID、作者、年份、DOI、来源 URL 和内容哈希；
- `labels_v1.json`：按问题和 PMID 生成的待标注模板。

如果输出目录已经存在冻结文件，脚本默认拒绝覆盖，以免误删人工标签；只有确认要重新冻结时才添加 `--force`。

## 2. 人工标注

打开 `labels_v1.json`，将每条 `relevance` 填为：

- `0`：不相关；
- `1`：部分相关；
- `2`：高度相关。

`coverage_scope` 固定为 `candidate_pool_only`，表示只对当前候选池标注，不能解释为 PubMed 全库 Recall。

## 3. 运行关键词基线

```powershell
python scripts/run_keyword_baseline.py `
  --snapshot data/benchmark/pubmed_snapshot_v1 `
  --output data/benchmark/keyword_baseline_v1.json
```

标签未全部填写时，程序会输出 `incomplete_labels`，不会生成虚假的总体指标。完成标注后，结果包含 Recall@1/3/5、MRR、nDCG、无结果率和平均延迟。

## 4. 使用本地交互式标注工具

不需要安装额外依赖，运行：

```powershell
python scripts/label_corpus.py `
  --snapshot data/benchmark/pubmed_snapshot_v1
```

工具会显示问题、PMID、标题和摘要，输入：

- `2`：高度相关；
- `1`：部分相关；
- `0`：不相关；
- `s`：跳过；
- `q`：保存并退出。

原始 `labels_v1.json` 不会被覆盖，标注结果默认保存到
`labels_v1_annotated.json`。中途退出后再次运行，会从未完成的条目继续。
