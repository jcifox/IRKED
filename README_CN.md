# IRKED：面向电信诈骗场景的 LLM 越狱防御

英文版说明见 `README.md`。

本项目是论文 **IRKED: Intent Recognition and Knowledge-Enhanced Defense against LLM Jailbreaks in Telecom-Fraud**对应的代码：

- **阶段 A：意图识别（Intent Recognition, IR）**：基于 `bert-base-chinese` 微调的二分类模型，判断输入是否为电诈/越狱相关恶意意图。
- **阶段 B：知识增强检测（Knowledge-Enhanced Detection, KE）**：当命中多个敏感实体后，触发语义检索，对照电诈知识库计算相似度并拦截伪装请求。

> 说明：仓库包含大文件与数据集，GitHub 不适合直接托管。请从下方 Google Drive 链接下载。

---

## 1. 概述

### 1.1 IRKED 两阶段防御主入口

- **`defense_integration1.py`**
  - `DefenseIntegratedLLM`：集成防御层 + 下游 LLM 调用
  - **模式**
    - `A`：仅 IR（BERT 意图识别）
    - `B`：仅 KE（实体 + 语义检索）
    - `C`：IR + KE（论文主设定）
  - **KE 组件**
    - `FraudKB`：加载 `fraud_kb.jsonl` 并用 `SentenceTransformer` 编码，运行期做余弦相似度检索
    - `build_ac_tree`：加载 `fraud_entity_dic.txt`，Aho–Corasick 多模式匹配（实体命中数不足 2 则直接放行）

### 1.2 下游大模型调用（论文“多模型端到端测试”）

- **`llm_api.py`**
  - `DomesticLLMClient`：统一封装 deepseek / qianwen / kimi / xinghuo / glm / chatgpt 的 API 调用
  - API Key 从 `.env` 或环境变量读取

### 1.3 IR（意图识别）训练与数据预处理

- **`data_collector.py`**
  - 从 `data/raw/` 聚合 CSV/JSON，生成 **大规模原始数据集** `data/raw_dataset_big.csv`
  - 同时包含“合成样本”逻辑（用于扩充样本量与覆盖多种电诈话术模式）

- **`data_preprocessor.py`**
  - 清洗/分词/特征扩展，输出 `data/processed_dataset_big.csv`

- **`train.py`**
  - 训练 IR（BERT 二分类）并保存模型到指定目录
  - 论文中的 `max_len=128`、`lr=2e-5`、`batch_size=16` 等超参可在此文件中对应设置

### 1.4 评测脚本与实验产物（论文 ISR/ASR、多轮与性能）

- **评测脚本（示例）**
  - `jailbreak_test.py`、`multiround_test.py`、`performance_test.py`
  - `advbench/`：用于通用基准（AdvBench）相关对比/测试的脚本集合

---

## 2. 目录结构速览

- **`data/`**
  - `raw/`：多源原始数据（体量大）
  - `raw_dataset_big.csv`：聚合后的大规模原始训练数据
  - `processed_dataset_big.csv`：预处理后的训练数据

- **`defense_models/robust_defense_best/`**
  - IR（BERT）微调后的模型文件（体量大）

- **`hf_models/`**
  - 本地缓存/镜像的 HuggingFace 模型（体量大）

- **`fraud_kb.jsonl`**
  - KE 阶段语义检索用的电诈知识库（JSON Lines：`{"text": "...", "label": 0/1}`）

- **`fraud_entity_dic.txt`**
  - KE 阶段敏感实体词典（Aho–Corasick）

- **`LLM-Sentry/`、`SelfDefend/`**
  - 相关对比方法的代码/数据
  - [LLM-Sentry](https://drive.google.com/drive/folders/10sM1kuWdImuqwkKhh5ZNrkGEat2pycu3?usp=sharing)
  - [SelfDefend](https://drive.google.com/drive/folders/10sM1kuWdImuqwkKhh5ZNrkGEat2pycu3?usp=sharing)

---

## 3. 数据与模型下载（Google Drive）

### 3.1 训练数据（`data.zip`）

[data.zip](https://drive.google.com/drive/folders/10sM1kuWdImuqwkKhh5ZNrkGEat2pycu3?usp=sharing) 包含：
- 训练数据：`data/raw_dataset_big.csv`、`data/processed_dataset_big.csv`
- 原始数据：`data/raw`

### 3.2 防御模型权重

[defense_models](https://drive.google.com/drive/folders/10sM1kuWdImuqwkKhh5ZNrkGEat2pycu3?usp=sharing)

### 3.3 本地 HuggingFace 模型镜像

[hf_models](https://drive.google.com/drive/folders/10sM1kuWdImuqwkKhh5ZNrkGEat2pycu3?usp=sharing) 包含：
- bert-base-chinese
- paraphrase-multilingual-MiniLM-L12-v2

