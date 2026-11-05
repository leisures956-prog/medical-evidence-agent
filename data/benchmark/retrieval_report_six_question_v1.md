# 固定候选池上的小规模人工检索评测报告

## 1. 评测概览

本报告评估关键词检索、Embedding 向量检索和关键词 + 向量混合检索在固定 PubMed 候选池上的排序表现。
评测日期为 2026-11-05。结果文件为
`retrieval_benchmark_six_question_v1.json`。

本报告的指标只适用于固定候选池，不代表 PubMed 全库召回率。

## 2. 数据与实验配置

| 项目 | 配置 |
| --- | --- |
| 语料版本 | `pubmed-snapshot-v1` |
| 候选范围 | `question_pool` |
| 问题总数 | 12 |
| 完整标注问题 | 6 |
| 未完成标注问题 | 6 |
| 每题候选文献 | 20 |
| 已标注记录 | 120 |
| 标注文件 | `labels_v1_six_question.json` |
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2` |
| 检索方法 | `keyword`、`vector`、`hybrid` |

完整标注的问题为：

```text
q01 lung cancer immunotherapy
q03 breast cancer screening
q05 diabetes GLP-1 receptor agonist
q07 acute ischemic stroke mechanical thrombectomy
q08 sepsis early diagnosis biomarkers
q12 chronic kidney disease SGLT2 inhibitors
```

另外 6 个问题仍未完成标注，因此没有参与 Recall、MRR 和 nDCG 的总体计算。

## 3. 标注与指标定义

人工标签定义如下：

- `0`：不相关；
- `1`：部分相关；
- `2`：高度相关。

Recall 和 MRR 将 `relevance >= 1` 视为相关文献；nDCG 使用 0/1/2 的分级相关性计算。
因此，Recall/MRR 反映是否找到了相关候选，nDCG 还反映高度相关文献是否排在更前面。

本次标注为单人标注，未计算标注者间一致性。结果应视为固定候选池上的小规模人工评测，
不能解释为 PubMed 全库的真实召回能力。

## 4. 检索结果

| 方法 | 完整问题 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@1 | nDCG@3 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword | 6 | 0.045 | 0.136 | 0.268 | 0.889 | 0.722 | 0.621 | 0.639 |
| Vector | 6 | 0.066 | 0.198 | 0.309 | 1.000 | 1.000 | 0.915 | 0.895 |
| Hybrid | 6 | 0.066 | 0.198 | 0.309 | 1.000 | 1.000 | 0.915 | 0.896 |

相较 Keyword，Vector 和 Hybrid 的 Recall@5 从 `0.268` 提升到 `0.309`，绝对提升约 4.0 个百分点。
Vector 和 Hybrid 的 Recall、MRR 基本相同；Hybrid 的 nDCG@5 略高于 Vector。

## 5. 延迟与索引构建

| 方法 | 平均查询延迟 | 中位数查询延迟 | P95 查询延迟 | 平均索引构建时间 |
| --- | ---: | ---: | ---: | ---: |
| Keyword | 2.04 ms | 2.00 ms | 2.90 ms | 0 ms |
| Vector | 21.91 ms | 13.99 ms | 91.20 ms | 676.80 ms |
| Hybrid | 17.17 ms | 16.46 ms | 20.90 ms | 676.80 ms |

Vector 和 Hybrid 需要构建 Embedding 索引；在本次运行中，Hybrid 的平均查询延迟和 P95 延迟低于
Vector。由于完整问题只有 6 个，延迟结果只作为当前机器和当前实现的观察值，不作为通用性能承诺。

## 6. 检索结果质量标记

质量标记来自标题、摘要是否存在和文章类型的透明启发式检查，不等同于医学质量金标准。
以下为 Top-5 中出现质量标记的比例：

| 方法 | 摘要缺失 | Editorial | Correction | Review | 任一质量标记 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keyword | 0.0% | 0.0% | 0.0% | 10.0% | 10.0% |
| Vector | 5.0% | 1.7% | 0.0% | 11.7% | 16.7% |
| Hybrid | 3.3% | 1.7% | 0.0% | 6.7% | 10.0% |

Hybrid 在 Top-5 的任一质量标记比例与 Keyword 相同，低于 Vector；但质量标记只用于提示人工复核，
不会自动判定文献无效。

## 7. Bad Case 与边界案例

### 7.1 乳腺癌筛查：主题相关但不是筛查研究

- 问题：`breast cancer screening`；Hybrid 排名：5；PMID：`42947695`；标签：`0`。
- 标题：`A nomogram integrating ultrasound features of the primary tumor and axillary lymph nodes with clinicopathological factors to predict high axillary lymph node burden in invasive breast cancer.`
- 分析：文献属于乳腺癌，但研究重点是腋窝淋巴结负荷预测，不是筛查。该案例说明仅依赖疾病词和语义相似度，可能把相关疾病研究误召回到筛查问题中。

### 7.2 卒中问题：疾病相关但方法不匹配

- 问题：`acute ischemic stroke mechanical thrombectomy`；Hybrid 排名：2；PMID：`42927926`；标签：`1`。
- 标题：`[Acute management of minor ischemic stroke].`
- 分析：文章讨论缺血性卒中急性处理，但标题没有直接涉及机械取栓，因此属于部分相关而非高度相关。后续可通过方法词过滤或重排降低此类边界召回。

### 7.3 SGLT2 与慢性肾病：研究范围部分重合

- 问题：`chronic kidney disease SGLT2 inhibitors`；Hybrid 排名：3；PMID：`42935914`；标签：`1`；质量标记：`review`。
- 标题：`SGLT2 Inhibitors for the Primary Prevention of CKD in Type 2 Diabetes: A Systematic Review and Meta-analysis.`
- 分析：文献同时涉及 SGLT2 和 CKD，但关注的是 2 型糖尿病人群中的一级预防，和一般慢性肾病治疗问题并不完全相同。综述标记提示使用者需要进一步查看摘要和研究范围。

### 7.4 质量标记案例

肺癌免疫治疗问题的 Hybrid Top-1 中出现 Editorial 且摘要缺失的文献；脓毒症问题的 Top-1 为 Review。
这些文献并非一定不相关，但不适合在没有摘要核验时直接作为强证据使用。

## 8. 结论

在 6 个完整标注问题、每题 20 篇固定候选文献的单人评测范围内：

1. Vector 和 Hybrid 的 Recall、MRR 和 nDCG 整体高于 Keyword；
2. Hybrid 与 Vector 的召回表现相近，nDCG@5 略高；
3. Hybrid 在本次运行中的查询延迟低于 Vector；
4. Hybrid Top-5 的质量标记比例为 10.0%，需要结合摘要缺失和综述标记进行人工复核；
5. 当前结果支持继续使用 Hybrid 作为本地候选排序方案，但不足以证明其在 PubMed 全库上普遍优于其他方法。

## 9. 局限性与后续工作

- 只有 6 个问题完成完整标注，样本规模较小；
- 标注来自单个标注者，未计算一致性和置信区间；
- 候选文献来自固定结果池，不能代表 PubMed 全库；
- 当前质量标记是启发式检查，不是医学事实或文献质量的自动判定；
- 评测没有进行统计显著性检验；
- 后续可以扩大问题数量、引入第二位标注者，并补充人工复核协议。

复现实验命令：

```powershell
.\.conda\search_api311\python.exe scripts\run_retrieval_benchmark.py `
  --snapshot data\benchmark\pubmed_snapshot_v1 `
  --labels data\benchmark\pubmed_snapshot_v1\labels_v1_six_question.json `
  --model "D:\ml-cache\models\paraphrase-multilingual-MiniLM-L12-v2" `
  --methods keyword vector hybrid `
  --candidate-scope question_pool `
  --output data\benchmark\retrieval_benchmark_six_question_v1.json
```
