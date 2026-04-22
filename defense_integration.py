# defense_integration.py  —— 知识增强版（兼容旧接口）
import os, json, random, logging, torch
import ahocorasick      # pip install pyahocorasick
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llm_api import DomesticLLMClient


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 1. 知识库检索器 ==========
class FraudKB:
    """离线编码知识库，运行期 cosine 检索"""
    def __init__(self, kb_path: str = "fraud_kb.jsonl", encoder_name: str = "hf_models/paraphrase-multilingual-MiniLM-L12-v2"):
        self.encoder = SentenceTransformer(encoder_name)
        self.data: List[str] = []
        self.vecs: np.ndarray = None
        if os.path.exists(kb_path):
            self._load_kb(kb_path)
        else:
            logger.warning(f"{kb_path} 不存在，知识库检索退化为空")

    def _load_kb(self, kb_path: str):
        texts = []
        errors = []
        try:
            with open(kb_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if item.get("label") == 1 and "text" in item:
                            texts.append(item["text"])
                        else:
                            continue
                    except json.JSONDecodeError as e:
                        errors.append(f"第 {line_num} 行 JSON 解析错误: {str(e)}")
                        logger.error(f"第 {line_num} 行 JSON 解析错误: {line[:100]}...")

            if texts:
                self.data = texts
                self.vecs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                logger.info(f"知识库加载完成，条目数：{len(texts)}")
                if errors:
                    logger.warning(f"知识库加载时有 {len(errors)} 个错误")
            else:
                logger.warning("知识库没有有效数据")

        except Exception as e:
            logger.error(f"加载知识库文件失败: {str(e)}")
            self.data = []
            self.vecs = None

    def search(self, query: str, top_k: int = 1) -> float:
        """返回最高 cosine 分数"""
        if not self.data or self.vecs is None:
            return 0.0
        try:
            q_vec = self.encoder.encode([query], normalize_embeddings=True)
            scores = (self.vecs @ q_vec.T).squeeze()
            return float(scores.max())
        except Exception as e:
            logger.error(f"知识库搜索失败: {str(e)}")
            return 0.0

# ========== 2. 实体词典 ==========
def build_ac_tree(dict_path: str = "fraud_entity_dic.txt"):
    tree = ahocorasick.Automaton()
    if not os.path.exists(dict_path):
        logger.warning(f"{dict_path} 不存在，实体检测退化为空")
        return tree
    try:
        with open(dict_path, encoding="utf-8") as f:
            for line_num, w in enumerate(f, 1):
                w = w.strip()
                if w:
                    tree.add_word(w.lower(), w)
        tree.make_automaton()
        logger.info(f"实体词典加载完成，词条数: {len(tree)}")
    except Exception as e:
        logger.error(f"加载实体词典失败: {str(e)}")
    return tree

# ========== 3. 知识增强检测器 ==========
class KADetector:
    def __init__(self, kb: FraudKB, ac_tree, threshold: float = 0.65):
        self.kb = kb
        self.ac_tree = ac_tree
        self.threshold = threshold

    def detect(self, text: str) -> bool:
        try:
            text_lower = text.lower()

            # 1. 实体命中
            entity_hits = list(self.ac_tree.iter(text_lower))
            if len(entity_hits) < 2:
                return False            # 实体不足 2，直接放行

            # 2.语义复核
            kb_score = self.kb.search(text)
            return kb_score >= self.threshold   # 同时满足才拦截
        except Exception as e:
            logger.error(f"知识增强检测失败: {str(e)}")
            return False

# ========== 4. 集成防御 ==========
class DefenseIntegratedLLM:
    def __init__(self,
                 defense_model_path: str = "defense_models/robust_defense_best",
                 llm_model_name: str = "deepseek",
                 llm_api_key: str = None,
                 llm_secret_key: str = None,
                 defense_mode: str = "C"):
        """
        初始化集成防御系统
        
        Args:
            defense_model_path: Bert防御模型路径
            llm_model_name: 大模型名称
            llm_api_key: API密钥
            llm_secret_key: 密钥
            defense_mode: 防御模式
                - 'A': 仅使用Bert检测模型
                - 'B': 仅使用知识增强（实体+语义检索）
                - 'C': 集成A和B的检测方式（默认）
        """
        # 验证防御模式
        if defense_mode not in ['A', 'B', 'C']:
            raise ValueError(f"不支持的防御模式: {defense_mode}，可选值: ['A', 'B', 'C']")
        self.defense_mode = defense_mode
        
        # 1. 原 Bert 检测器（所有模式都需要初始化，但根据模式可能不启用）
        if defense_mode in ['A', 'C']:
            try:
                self.defense_tokenizer = AutoTokenizer.from_pretrained(defense_model_path)
                self.defense_model = AutoModelForSequenceClassification.from_pretrained(defense_model_path)
                self.defense_model.eval()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.defense_model.to(self.device)
                logger.info(f"Bert防御模型加载完成，设备: {self.device}")
            except Exception as e:
                logger.error(f"加载Bert防御模型失败: {str(e)}")
                self.defense_tokenizer = None
                self.defense_model = None
                self.device = "cpu"
        else:
            self.defense_tokenizer = None
            self.defense_model = None
            self.device = "cpu"

        # 2. 知识增强检测器（所有模式都需要初始化，但根据模式可能不启用）
        if defense_mode in ['B', 'C']:
            try:
                self.kb = FraudKB()
                self.ac_tree = build_ac_tree()
                self.ka_detector = KADetector(self.kb, self.ac_tree)
                logger.info(f"知识增强检测器加载完成")
            except Exception as e:
                logger.error(f"加载知识增强检测器失败: {str(e)}")
                self.kb = None
                self.ac_tree = None
                self.ka_detector = None
        else:
            self.kb = None
            self.ac_tree = None
            self.ka_detector = None

        # 3. 大模型客户端
        try:
            self.llm_client = DomesticLLMClient(
                model_name=llm_model_name,
                api_key=llm_api_key or os.getenv(f"{llm_model_name.upper()}_API_KEY"),
                secret_key=llm_secret_key or os.getenv(f"{llm_model_name.upper()}_SECRET_KEY")
            )
            logger.info(f"大模型客户端加载完成: {llm_model_name}")
        except Exception as e:
            logger.error(f"加载大模型客户端失败: {str(e)}")
            raise e

        # 4. 安全模板
        self.safe_responses = [
            "您的请求涉及非法内容，无法提供帮助。",
            "该内容可能违反法律法规，建议停止此类请求。",
            "抱歉，您的输入包含敏感信息，不支持处理。"
        ]
        
        logger.info(f"防御系统初始化完成，模式: {defense_mode}")

    # ---- 原接口保持不变 ----
    def process_input(self, user_input: str, bypass_defense: bool = False):
        """处理用户输入，根据防御模式进行检测"""
        if not bypass_defense:
            # 根据防御模式选择检测策略
            if self.defense_mode == 'A':
                # 仅使用Bert检测
                if self._bert_detect(user_input):
                    return random.choice(self.safe_responses)
                    
            elif self.defense_mode == 'B':
                # 仅使用知识增强检测
                if self._ka_detect(user_input):
                    return random.choice(self.safe_responses)
                    
            elif self.defense_mode == 'C':
                # 集成检测（Bert + 知识增强）
                bert_flag = self._bert_detect(user_input)
                ka_flag = self._ka_detect(user_input)
                if bert_flag or ka_flag:
                    return random.choice(self.safe_responses)
        
        # 调用大模型生成回复
        try:
            return self.llm_client.generate(prompt=user_input)
        except Exception as e:
            logger.error(f"大模型生成失败: {str(e)}")
            return "抱歉，系统处理您的请求时出现错误。"

    # ---- Bert检测函数 ----
    def _bert_detect(self, user_input: str, threshold: float = 0.5) -> bool:
        """Bert恶意内容检测"""
        if self.defense_mode not in ['A', 'C'] or not self.defense_model:
            return False
            
        try:
            encoding = self.defense_tokenizer(
                user_input, truncation=True, padding=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.defense_model(**encoding).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return probs[1] > threshold
        except Exception as e:
            logger.error(f"Bert检测失败: {e}")
            return False

    # ---- 知识增强检测函数 ----
    def _ka_detect(self, user_input: str) -> bool:
        """知识增强检测"""
        if self.defense_mode not in ['B', 'C'] or not self.ka_detector:
            return False
            
        try:
            return self.ka_detector.detect(user_input)
        except Exception as e:
            logger.error(f"知识增强检测失败: {e}")
            return False

    # ---- 向后兼容的检测接口（保持原有测试代码兼容性） ----
    def _detect_malicious(self, user_input: str, threshold: float = 0.5):
        """向后兼容接口，返回详细检测结果"""
        if self.defense_mode not in ['A', 'C'] or not self.defense_model:
            return {"is_malicious": False, "confidence": 0.0}
            
        try:
            encoding = self.defense_tokenizer(
                user_input, truncation=True, padding=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.defense_model(**encoding).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            is_malicious = probs[1] > threshold
            return {"is_malicious": is_malicious, "confidence": probs[1]}
        except Exception as e:
            logger.error(f"Bert检测失败: {e}")
            return {"is_malicious": False, "confidence": 0.0}