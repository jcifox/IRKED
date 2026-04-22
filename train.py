# train.py
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import torch
import torch.nn as nn
import warnings
import shutil
# >>> 新增：TensorBoard & 画图
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
# <<<

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["WANDB_SILENT"] = "true"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------- Focal Loss --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ------------------ FocalLossTrainer ------------------
class FocalLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.focal_loss = kwargs.pop('focal_loss', None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------- DefenseModelTrainer ----------------
class DefenseModelTrainer:
    def __init__(self, model_path="hf_models/bert-base-chinese",
                 save_dir="defense_models111/robust_defense_best"):
        self.model_path = model_path
        self.save_dir   = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.tokenizer  = BertTokenizer.from_pretrained(self.model_path)
        # TensorBoard 目录按时间区分
        self.tb_writer  = SummaryWriter(
            log_dir=f"runs/defense_{pd.Timestamp.now().strftime('%m%d_%H%M')}"
        )
        logger.info("TensorBoard 记录目录: %s", self.tb_writer.log_dir)

    # ------------------ 内部工具函数 ------------------
    def _preprocess_text(self, examples):
        return self.tokenizer(
            examples["cleaned_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro"),
            "recall": recall_score(labels, preds, average="macro"),
            "f1": f1_score(labels, preds, average="macro"),
            "auc": roc_auc_score(labels, logits[:, 1])
        }

    def _get_tb_callback(self):
        from transformers import TrainerCallback
        class TBCallback(TrainerCallback):
            def __init__(self, writer):
                super().__init__()
                self.writer = writer
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    for k, v in logs.items():
                        if k in {'loss', 'eval_loss', 'learning_rate'}:
                            self.writer.add_scalar(k, v, state.global_step)
        return TBCallback(self.tb_writer)

    # ------------------ 画图 ------------------
    def _plot_loss(self, trainer):
        history = trainer.state.log_history
        train_loss, eval_loss, steps = [], [], []
        for d in history:
            if 'loss' in d and 'step' in d:
                train_loss.append(d['loss'])
                steps.append(d['step'])
            if 'eval_loss' in d and 'step' in d:
                eval_loss.append(d['eval_loss'])

        plt.figure(figsize=(6, 4))
        if train_loss:
            plt.plot(steps[:len(train_loss)], train_loss, label='Train Loss')
        if eval_loss:
            plt.plot(steps[:len(eval_loss)], eval_loss, label='Eval Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_pdf = os.path.join(self.tb_writer.log_dir, 'loss_curve.pdf')
        out_png = os.path.join(self.tb_writer.log_dir, 'loss_curve.png')
        plt.savefig(out_pdf, format='pdf')
        plt.savefig(out_png, dpi=300)
        logger.info("高清损失曲线已保存至 %s / %s", out_pdf, out_png)
        self.tb_writer.add_figure('loss/curve', plt.gcf())
        plt.close()

    # ------------------ 训练主入口 ------------------
    def train_model(self, train_df, epochs=10, learning_rate=2e-5,
                    batch_size=16, weight_decay=0.01,
                    focal_loss_params=None):
        if focal_loss_params is None:
            focal_loss_params = {"alpha": 0.8, "gamma": 2.0}

        # 1. 清洗数据
        train_df = train_df.dropna(subset=["label"]).reset_index(drop=True)
        train_df = train_df.fillna(0)
        train_df["label"] = train_df["label"].astype(np.int64)
        logger.info("清洗后样本数：%d", len(train_df))
        logger.info("正负样本分布：%s", train_df.label.value_counts().to_dict())

        # 2. 拆分
        train_data, val_data = train_test_split(
            train_df[["cleaned_text", "label"]],
            test_size=0.3, random_state=42, stratify=train_df["label"]
        )
        logger.info("训练集：%d 验证集：%d", len(train_data), len(val_data))

        # 3. Dataset
        train_dataset = Dataset.from_pandas(train_data).map(self._preprocess_text, batched=True)
        val_dataset   = Dataset.from_pandas(val_data).map(self._preprocess_text, batched=True)
        for ds in [train_dataset, val_dataset]:
            ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # 4. 模型
        model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)

        # 5. TrainingArguments
        training_args = TrainingArguments(
            output_dir="./temp_defense_train",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            logging_dir="./defense_logs",
            logging_steps=250,
            report_to="none",
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available()
        )

        # 6. FocalLoss & Trainer
        focal_loss = FocalLoss(**focal_loss_params)
        trainer = FocalLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            focal_loss=focal_loss,
            callbacks=[self._get_tb_callback()]
        )

        # 7. 训练
        logger.info("开始训练...")
        trainer.train()

        # 8. 评估
        logger.info("评估最优模型...")
        predictions, labels, _ = trainer.predict(val_dataset)
        preds = np.argmax(predictions, axis=-1)
        print("\n===== 最优模型评估结果 =====")
        print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
        print(f"Precision: {precision_score(labels, preds, average='macro'):.4f}")
        print(f"Recall: {recall_score(labels, preds, average='macro'):.4f}")
        print(f"Macro-F1: {f1_score(labels, preds, average='macro'):.4f}")
        print(f"AUC: {roc_auc_score(labels, predictions[:, 1]):.4f}")
        print("\n分类报告:")
        print(classification_report(labels, preds, target_names=["良性", "恶意"]))
        print("\n混淆矩阵:")
        print(confusion_matrix(labels, preds))

        # 9. 保存模型
        model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        logger.info("最优模型已保存至 %s", self.save_dir)

        # 10. 清理
        if os.path.exists("./temp_defense_train"):
            shutil.rmtree("./temp_defense_train")
            logger.info("已清理训练临时文件")

        # 11. 画损失曲线
        self._plot_loss(trainer)
        self.tb_writer.close()
        return model


# ------------------- main -------------------
if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed_dataset_big.csv"
    MODEL_SAVE_DIR      = "defense_models111/robust_defense_best"

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"请先处理数据集并生成 {PROCESSED_DATA_PATH}")

    defense_data = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info("已加载预处理数据，共 %d 条记录", len(defense_data))

    trainer = DefenseModelTrainer(save_dir=MODEL_SAVE_DIR)
    trainer.train_model(
        train_df=defense_data,
        epochs=10,
        learning_rate=2e-5,
        batch_size=16,
        weight_decay=0.01,
        focal_loss_params={"alpha": 0.8, "gamma": 2.0}
    )