import re
import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import chardet  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """预处理文本数据，用于模型训练和测试，支持大规模数据集处理"""

    def __init__(self):
        # 扩展敏感词列表
        self.sensitive_words = {
            "诈骗相关": ["诈骗", "欺诈", "骗子", "行骗", "骗取", "scam", "fraud"],
            "金融操作": ["转账", "汇款", "银行卡", "验证码", "账户", "套现", "支付", "充值", "提现"],
            "身份伪装": ["冒充", "假扮", "伪装", "假装", "伪造", "克隆", "仿冒"],
            "诱导行为": ["诱导", "说服", "让", "使", "配合", "引导", "怂恿", "诱骗"]
        }
        # 扩展停用词表
        self.stopwords = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
                             "可以", "为", "以", "让", "用", "被", "于", "对", "将", "从", "到",
                             "之", "与", "等", "更", "也", "很", "还", "要", "会", "能", "好"])

    # -------------------- 编码检测 --------------------
    def _detect_encoding(self, file_path):
        """检测文件编码，支持多种常见编码"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        
        # 检测可能的编码
        result = chardet.detect(raw_data)
        candidate_encodings = [
            result['encoding'], 
            'utf-8', 
            'gbk', 
            'gb2312', 
            'utf-16',
            'iso-8859-1'
        ]
        
        # 尝试每个候选编码
        for encoding in candidate_encodings:
            if encoding:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # 尝试读取部分内容
                    return encoding
                except (UnicodeDecodeError, LookupError):
                    continue
        
        # 如果所有尝试都失败，返回默认编码
        return 'utf-8'

    # -------------------- 文本清洗 --------------------
    def _clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # 只保留中英文字符和数字
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        # 分词并过滤停用词和短词
        words = [w for w in jieba.cut(text.strip()) if w not in self.stopwords and len(w) > 1]
        return " ".join(words)

    # -------------------- 特征工程 --------------------
    def _extract_features(self, text: str) -> dict:
        features = {}
        features["text_length"] = len(text)
        features["word_count"] = len(text.split()) if text else 0
        
        # 敏感词特征（扩展版）
        for category, words in self.sensitive_words.items():
            features[f"{category}_count"] = sum(1 for w in words if w in text)
        
        # 攻击类型关键词特征
        attack_keywords = {
            "角色扮演伪装": ["假设", "扮演", "模拟", "写小说", "剧本", "冒充", "假扮", "伪装成", "假装成"],
            "教育研究伪装": ["研究", "教学", "案例", "课程", "学术", "论文", "教材", "测试", "题库"],
            "模糊指令": ["怎么", "如何", "办法", "技巧", "有什么", "怎样", "哪些", "如何才能"],
            "多步诱导": ["第一步", "后续", "接下来", "先", "后面", "下一步", "分阶段", "逐步"],
            "技术工具滥用": ["软件", "脚本", "接口", "AI", "克隆", "伪造", "钓鱼网站", "伪基站", 
                          "嗅探", "改号", "换脸", "生成器", "批量", "自动化", "第三方", 
                          "调用", "搭建", "服务器", "硬件", "连接图", "实时更新", "余额", 
                          "假银行", "App", "设备", "程序", "工具"],
            "套现诈骗": ["套现", "套取现金", "提现", "兑现", "POS机", "信用卡"],
            "转账诈骗": ["转账", "汇款", "打款", "划款", "转钱", "安全账户", "验证码"],
            "电信诈骗": ["电话", "短信", "客服", "中心", "10086", "10010", "95588"],
            "公检法诈骗": ["公检法", "法院", "检察院", "公安局", "涉嫌", "洗钱", "案件"],
            "钓鱼诈骗": ["钓鱼", "链接", "网址", "网站", "登录", "账号", "密码"],
            "虚假信息诈骗": ["虚假", "伪造", "假的", "不实", "冒充", "谎称"],
            "集资诈骗": ["集资", "投资", "理财", "高息", "回报", "收益", "项目"],
            "新型诈骗": ["AI", "元宇宙", "虚拟资产", "区块链", "加密货币", "ChatGPT"]
        }
        
        for attack_type, keywords in attack_keywords.items():
            features[f"has_{attack_type}_kw"] = int(any(k in text for k in keywords))
            
        # 新增：关键词密度特征
        total_words = features["word_count"]
        if total_words > 0:
            for category, words in self.sensitive_words.items():
                count = sum(1 for w in words if w in text)
                features[f"{category}_density"] = count / total_words
        
        return features

    # -------------------- 核心接口 --------------------
    def preprocess_data(self, raw_df: pd.DataFrame, min_len: int = 5) -> pd.DataFrame:
        """
        预处理数据，增加对大规模数据集的支持
        
        参数:
            raw_df: 原始数据DataFrame
            min_len: 最小文本长度阈值，过滤过短文本
        """
        logger.info(f"原始数据 shape: {raw_df.shape}")
        logger.info(f"原始 label 分布:\n{raw_df.label.value_counts().sort_index().to_dict()}")
        
        # 处理可能的缺失值
        df = raw_df.copy().dropna(subset=["text", "label"])
        
        # 1. 清洗文本
        logger.info("开始清洗文本数据...")
        df["cleaned_text"] = df["text"].apply(self._clean_text)
        
        # 2. 过滤短文本
        before = len(df)
        df = df[df["cleaned_text"].str.len() >= min_len].copy()
        after = len(df)
        logger.info(f"文本长度 < {min_len} 丢弃 {before - after} 条，剩余 {after} 条")
        logger.info(f"过滤后 label 分布:\n{df.label.value_counts().sort_index().to_dict()}")
        
        # 3. 提取特征（分批处理，适合大规模数据）
        logger.info("开始提取特征...")
        batch_size = 10000  # 批处理大小
        all_features = []
        
        for i in range(0, len(df), batch_size):
            batch = df["cleaned_text"].iloc[i:i+batch_size]
            features = batch.apply(self._extract_features)
            all_features.extend(features.tolist())
            
            # 打印进度
            if (i // batch_size) % 10 == 0:
                logger.info(f"特征提取进度: {min(i+batch_size, len(df))}/{len(df)}")
        
        features_df = pd.DataFrame(all_features)
        df = pd.concat([df, features_df], axis=1)
        
        # 4. 保存处理后的数据
        os.makedirs("data", exist_ok=True)
        out_path = "data/processed_dataset_big.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"预处理完成，已保存至 {out_path}")
        logger.info("新增特征列: %s", [c for c in df.columns if c not in {"text", "label", "attack_type"}])
        
        return df

    # -------------------- 加载数据 --------------------
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据，支持自动检测编码"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到原始数据文件：{file_path}")
            
        logger.info(f"开始加载数据: {file_path}")
        encoding = self._detect_encoding(file_path)
        logger.info(f"检测到文件编码: {encoding}")
        
        # 对于大型CSV，使用分块读取
        chunk_size = 50000
        chunks = []
        for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size):
            # 确保必要的列存在
            required_cols = ["text", "label", "attack_type"]
            for col in required_cols:
                if col not in chunk.columns:
                    chunk[col] = "" if col != "label" else 0
            chunks.append(chunk)
            logger.info(f"已加载 {len(chunks)*chunk_size} 条数据")
            
        raw_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"数据加载完成，共 {len(raw_df)} 条")
        return raw_df


# -------------------- CLI --------------------
if __name__ == "__main__":
    # 处理大规模数据集
    RAW_PATH = "data/raw_dataset_big.csv"
    preprocessor = DataPreprocessor()
    
    try:
        raw_df = preprocessor.load_raw_data(RAW_PATH)
        # 对于大规模数据，适当提高最小长度阈值
        preprocessor.preprocess_data(raw_df, min_len=5)
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}", exc_info=True)