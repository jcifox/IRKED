# IRKED: Intent Recognition and Knowledge-Enhanced Defense

Chinese version: `README_CN.md`.

This repository accompanies **IRKED: Intent Recognition and Knowledge-Enhanced Defense against LLM Jailbreaks in Telecom-Fraud** . It contains the implementation and experiment artifacts for a **two-stage, pre-invocation, model-agnostic defense pipeline**:

- **Stage A — Intent Recognition (IR)**: a binary classifier fine-tuned from `bert-base-chinese` to identify telecom-fraud / jailbreak intent from the prompt.
- **Stage B — Knowledge-Enhanced Detection (KE)**: triggers on multiple sensitive-entity hits, then performs semantic retrieval against a fraud knowledge base and blocks disguised high-risk requests.

> Note: the project contains large datasets and model files that are not suitable for GitHub. Please download them from Google Drive (links below).

---

## 1. Overview

### 1.1 Key implementation files

- **`defense_integration.py`**
  - `DefenseIntegratedLLM`: the integrated defense layer + downstream LLM invocation
  - **Modes**
    - `A`: IR only (BERT intent classifier)
    - `B`: KE only (entity trigger + semantic retrieval)
    - `C`: IR + KE (paper default)
  - **KE components**
    - `FraudKB`: loads `fraud_kb.jsonl` and encodes entries with `SentenceTransformer`, then retrieves by cosine similarity
    - `build_ac_tree`: loads `fraud_entity_dic.txt` and builds an Aho–Corasick automaton (passes through if entity hits < 2)

### 1.2 Downstream LLM invocation

- **`llm_api.py`**
  - `DomesticLLMClient`: a unified API client for deepseek / qianwen / kimi / xinghuo / glm / chatgpt
  - API keys are read from `.env` or environment variables

### 1.3 Training and data preparation

- **`data_collector.py`**
  - Aggregates CSV/JSON files under `data/raw/` and generates the large-scale dataset `data/raw_dataset_big.csv`
  - Includes synthetic-sample generation logic to expand coverage

- **`data_preprocessor.py`**
  - Cleans text, tokenizes, and performs feature expansion; outputs `data/processed_dataset_big.csv`

- **`train.py`**
  - Trains the IR model (BERT binary classifier) and saves the best checkpoint
  - Paper hyperparameters (e.g., `max_len=128`, `lr=2e-5`, `batch_size=16`) can be configured here

### 1.4 Evaluation and artifacts

- **Evaluation scripts (examples)**
  - `jailbreak_test.py`, `multiround_test.py`, `performance_test.py`
  - `advbench/`: scripts for the general benchmark (AdvBench) comparisons/tests

---

## 2. Directory overview

- **`data/`**
  - `raw/`: multi-source raw data (large)
  - `raw_dataset_big.csv`: aggregated large-scale raw training dataset
  - `processed_dataset_big.csv`: preprocessed training dataset

- **`defense_models/robust_defense_best/`**
  - Fine-tuned IR (BERT) model files (large)

- **`hf_models/`**
  - Local HuggingFace model mirrors (large)

- **`fraud_kb.jsonl`**
  - Fraud knowledge base for KE semantic retrieval (JSON Lines: `{"text": "...", "label": 0/1}`)

- **`fraud_entity_dic.txt`**
  - Sensitive-entity dictionary for KE (Aho–Corasick)

- **`LLM-Sentry/`, `SelfDefend/`**
  - Baseline methods (code/data); 
  - [LLM-Sentry](https://drive.google.com/file/d/1xWG9Wuus9hA3pvgniC0qT3GG9aBpPpzp/view?usp=sharing)
  - [SelfDefend](https://drive.google.com/file/d/144y1MyG-BfxJhZTq9e74IpRUn4Qi5lvs/view?usp=sharing)

---

## 3. Data and models

Click the links below to download.

### 3.1 Training data (`data.zip`)

[data.zip](https://drive.google.com/file/d/1sBTQP0KHIpRIvVxa_6L3vVie_zpmGVQj/view?usp=sharing) includes:
- Training data: `data/raw_dataset_big.csv`, `data/processed_dataset_big.csv`
- Raw data: `data/raw`

[defense_models](https://drive.google.com/file/d/1-0kO4CqZa11d5R1lqqeoqok_WTytbHGY/view?usp=sharing)

[hf_models](https://drive.google.com/file/d/11W-CE0GeWpr9i1vtw7BmA4Qa7FnMKwQ0/view?usp=sharing) includes:
- bert-base-chinese
- paraphrase-multilingual-MiniLM-L12-v2
