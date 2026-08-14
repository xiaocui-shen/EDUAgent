# backend/core/query_classifier.py

import json
import os
import torch
import random
from pathlib import Path
from typing import Any, Optional

from backend.config import get_settings # 配置单例
from backend.config import get_abs_path # 基于项目目录获取绝对路径
from backend.core.logger import get_logger
logger = get_logger(__name__) # 模块级日志器

# general 侧置信度阈值设为 0.85，有点偏高、具体原因：
# 专业问题被误判为通用问题的代价更高——LLM 会用自身知识回答，
# 可能与课程内容矛盾。宁可多走一次 RAG，不要漏掉课程相关问题。
GENERAL_CONFIDENCE_THRESHOLD = 0.85
class QueryClassifier:
    """
    QA Query 二分类器：general / specialized。
    训练阶段：
        qc = QueryClassifier(True)
        qc.train("tmp/training_data.jsonl",
                 output_dir="models/classifier")
    推理阶段：
        qc = QueryClassifier()
        label, conf = qc.classify("什么是 Spring IOC？")
    """

    _instance: Optional["QueryClassifier"] = None

    def __init__(self, train_model: bool = False):
        """
        Args:
            train_model: 是否训练模型。
                默认False 加载微调模型，直接用于推理。
                True 加载基座 MiniLM，用于训练。
        """
        if train_model: # 训练模式
            # 从配置读取本地基座模型路径 （用于训练）
            self.model_path = get_abs_path(
                get_settings().classifier_model_path)
            # 微调模型保存目录
            self.output_dir = get_abs_path(
                get_settings().classifier_finetuned_model_path)
            logger.info("query_classifier.training")
        else: # 推理模式
            # 微调训练后的意图分类模型路径
            self.model_path = get_abs_path(
                get_settings().classifier_finetuned_model_path)
            logger.info("query_classifier.using")
        # 自动检测设备：CUDA GPU 加速 或 CPU
        device = 0 if torch.cuda.is_available() else -1
        # 加载文本分类模型
        from transformers import pipeline as hf_pipeline
        self._pipeline: Any = hf_pipeline(
            task="text-classification", # 文本分类任务
            model=self.model_path, # 模型绝对路径
            device=device, # 0=cuda:0, -1=cpu
            top_k=None, # 返回全部标签分数，而非只返回 top-1
            truncation=True, # 超出 max_length 自动截断
            max_length=128, # token 最大长度，Query 分类用不到长文本
            # model_kwargs参数：绕过 SDPA 的 meta tensor bug
            model_kwargs={"attn_implementation": "eager"},
        )
        '''model_kwargs={"attn_implementation": "eager"}
        transformers4.40+较新版本 默认用SDPA实现
            （SDPA：Scaled Dot Product Attention）
        但在CPU上（device=-1）
            torch的SDPA路径会把attention mask放到 meta device上
            导致 torch.all(mask == 1) 内部调 .item() 时出错
        '''
        logger.info("query_classifier.loading",
                    model_path=self.model_path,
                    device=device)

    @classmethod
    def get_instance(cls) -> "QueryClassifier":
        """获取单例（首次调用时懒加载）"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 训练 ─────────────────────────────────────────────────

    def train(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 2e-5,
        max_length: int = 128,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        微调当前加载的基座模型，训练完自动保存到 output_dir。

        Args:
            data_path:  训练数据路径（JSONL，每行 {"text": ..., "label": "general/specialized"}）
            output_dir: 微调模型保存目录（训练完可直接用此路径初始化新实例）
            epochs:     训练轮数（默认 8）
            batch_size: 训练批大小（默认 64）
            lr:         学习率（默认 2e-5）
            max_length: Token 最大长度（默认 128，Query 分类用不到长文本）
            val_ratio:  验证集比例（默认 0.1）
            test_ratio: 测试集比例（默认 0.1）
            seed:       随机种子（默认 42）
        """
        # 未传 output_dir 时，默认使用 __init__ 中配置的保存路径
        output_dir = output_dir or self.output_dir
        # 训练库仅在此方法内 import，推理路径不加载这些依赖
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        # 设置随机种子：random、NumPy、PyTorch的随机都设置为seed
        set_seed(seed)

        # ── 标签映射：字符串 ↔ 整数 ──
        #   - general：公开可查的 IT 概念、原理、语法，LLM 直接回答
        #   - specialized：涉及具体课程/实战项目内容，必须检索知识库
        # 标签 → 模型输出ID
        LABEL2ID = {"general": 0, "specialized": 1}
        # 模型输出ID → 标签
        ID2LABEL = {0: "general", 1: "specialized"}
        # 创建 Tokenizer 词转token ID，用于将文本转换为模型输入

        # ── Tokenizer + Model ─────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True) # fast tokenizer 速度更快
        # 创建 Model 模型，用于训练
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=2, # 二分类
            label2id=LABEL2ID, # 标签名→ID映射
            id2label=ID2LABEL, # ID→标签名映射
            ignore_mismatched_sizes=True, # 分类头从2类随机初始化
        )

        # ── 加载数据 ──────────────────────────────────────────
        rows = self._load_jsonl(data_path)
        logger.info(
            "query_classifier.train_data_loaded",
            total=len(rows),
            general=sum(
                1 for r in rows if r["label"] == "general"),
            specialized=sum(
                1 for r in rows if r["label"] == "specialized")
        )
        # 数据集切分，按比例划分训练集、验证集和测试集
        train_rows, val_rows, test_rows = self._stratified_split(
            rows,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )

        # ── 数据集处理 ────────────────────────────────────────
        def to_dataset(rows_: list[dict]) -> Dataset:
            """将行列表转为 HuggingFace Dataset"""
            return Dataset.from_dict({
                "text":  [r["text"] for r in rows_],
                "label": [LABEL2ID[r["label"]]
                          for r in rows_], # 标签转整数
            })
        def tokenize(batch):
            """对批次做 tokenize，截断到 max_length"""
            return tokenizer(batch["text"],
                             truncation=True,
                             max_length=max_length)
        # 构建三个数据集（文本→tokenized）
        train_ds = to_dataset(train_rows).map(
            tokenize, batched=True, remove_columns=["text"])
        val_ds = to_dataset(val_rows).map(
            tokenize,   batched=True, remove_columns=["text"])
        test_ds = to_dataset(test_rows).map(
            tokenize,  batched=True, remove_columns=["text"])

        logger.info("query_classifier.split",
                    train=len(train_rows),
                    val=len(val_rows),
                    test=len(test_rows))

        # ── 评估指标 ──────────────────────────────────────────
        # Trainer 每个 epoch 结束时调用此函数评估模型在验证集上的表现，
        # 返回值用于早停判断（patience=2 即连续2轮F1不提升则停止训练）
        def compute_metrics(eval_pred):
            """计算 accuracy（准确率）和 macro F1（宏平均F1分数）。
            accuracy: 预测正确的样本数 / 总样本数，反映整体分类正确率
            macro F1: 对每个类别单独算F1再取平均，不受类别不平衡影响，
                      比accuracy更能反映小样本类别的分类质量
            """
            # 解包：logits是模型输出的原始分数（未经过softmax），
            # shape=(batch_size, num_labels)，值越大表示模型越倾向该类别
            # labels是真实标签，shape=(batch_size,)
            logits, labels = eval_pred
            # argmax沿axis=-1（最后一个维度，即类别维度）取最大值的索引，
            # 将原始分数转为预测的类别ID（0=general, 1=specialized）
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds) # 计算整体准确率
            # precision_recall_fscore_support返回4个值：
            # precision(精确率), recall(召回率), f1, support(每类样本数)
            # 只取f1：精确率和召回率的调和平均，综合衡量分类质量
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, # 真实标签 和 预测标签
                # 宏平均：每个类别权重相同，不受样本数影响
                average="macro",
                # 某类完全没有预测样本时返回0，避免除零报错
                zero_division=0
            )
            # 返回acc整体准确率和宏平均F1
            # Trainer会根据metric_for_best_model="f1_macro"
            #   从而选出F1最高的checkpoint作为最优模型
            return {"accuracy": float(acc), "f1_macro": float(f1)}

        # ── TrainingArguments 训练参数 ───────────────────────
        use_cuda = torch.cuda.is_available() # 是否可用GPU
        # 中间checkpoint目录
        checkpoint_dir = str(Path(output_dir) / "_checkpoints")
        # 训练参数对象
        train_args = TrainingArguments(
            output_dir=checkpoint_dir, # checkpoint保存路径
            eval_strategy="epoch", # 每个epoch结束时评估
            save_strategy="epoch", # 每个epoch结束时保存
            # 训练结束自动加载最优checkpoint
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro", # 按F1选最优
            greater_is_better=True, # F1越大越好
            num_train_epochs=epochs, # 训练轮数
            per_device_train_batch_size=batch_size, # 训练批大小
            # 评估时批大小翻倍效率更高（评估不需要反向传播，内存/显存占用更小）
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr, # 学习率，默认2e-5
            warmup_ratio=0.1, # 前10%步数线性预热，避免初期震荡
            weight_decay=0.01, # L2正则化，防止过拟合
            save_total_limit=2, # 最多保留2个checkpoint，省磁盘
            logging_steps=20, # 每20步输出一次训练日志
            fp16=use_cuda, # 如果启用GPU就使用半精度加速
            report_to="none", # 不上报第三方平台
            seed=seed, # 固定随机种子
        )

        # 构造 Trainer：封装模型、数据、训练参数
        trainer = Trainer(
            model=model, # 预训练模型
            args=train_args, # 训练参数
            train_dataset=train_ds, # 训练集
            eval_dataset=val_ds, # 验证集（每个epoch评估）
            tokenizer=tokenizer, # 分词器
            # 动态padding，按批次最长序列对齐
            # 训练时模型要求同一批次里所有句子长度必须一样
            # 但实际每个 query 长度不同
            # data_collator：把批次里所有句子动态补齐到最长的那个的长度
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics, # 评估指标函数
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2)
            ], # 连续2个epoch不提升即早停
        )

        # ── 训练 + 测试集评估 ──────────────────────────────────
        trainer.train() # 开始训练
        # 在测试集上评估，返回test_metrics测试结果字典
        test_metrics = trainer.evaluate(test_ds)
        logger.info( # 记录eval_*指标
            "query_classifier.test_metrics",
            **{k: round(v, 4)
               for k, v in test_metrics.items() # 遍历测试结果字典
               # 给k指标名添加前缀 eval_，
               if k.startswith("eval_")
        })

        # ── 保存训练好的模型 ────────────────────────────────────
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir) # 保存模型权重+配置
        tokenizer.save_pretrained(output_dir) # 保存tokenizer
        logger.info("query_classifier.saved",
                    output_dir=output_dir)
        print(f"\n✅ 微调完成，模型保存到：{output_dir}")
        print(f"   下次使用：QueryClassifier()")

    # ── 推理 ─────────────────────────────────────────────────
    def classify(self, text: str) -> tuple[str, float]:
        """
        对 query 做 general / specialized 二分类。
        Returns:
            (label, confidence)
            label:      "general" 或 "specialized"
            confidence: 对应标签的置信度 [0, 1]
        分类规则：
            P(general) >= 0.85 → ("general",      P(general))
            其余               → ("specialized",  1 - P(general))
        兜底：标签名不匹配时保守返回 ("specialized", 0.5)。
        """
        # 执行分类推理，返回分类名和对应的概率结果分数
        '''[
            {"label": "general",     "score": 0.12},
            {"label": "specialized", "score": 0.88},
        ]'''
        raw_outputs: list[dict] = self._pipeline(text)[0]

        # 从pipeline结果中找general标签的分数
        general_score = None
        for item in raw_outputs:
            if item["label"] == "general":
                general_score = item["score"]
                break

        # 分类决策：P(general) >= 阈值 → general，否则 → specialized
        if general_score >= GENERAL_CONFIDENCE_THRESHOLD:
            # general 通用问题（后续LLM直接回答）
            label, confidence = "general", general_score
        else: # specialized 专业问题（后续需要RAG）
            label, confidence = "specialized", 1.0 - general_score

        # 返回分类结果标签 和 该分类的置信度
        logger.info("query_classifier.result",
                    text_preview=text[:50],
                    label=label,
                    confidence=round(confidence, 4))
        return label, confidence

    # ── 私有工具方法 ─────────────────────────────────────────
    # 校验训练数据
    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        """从 JSONL文件读取训练数据，逐行校验 text 和 label"""
        LABEL2ID = {"general": 0, "specialized": 1} # 标签
        rows: list[dict] = [] # 最终返回的数据列表
        with open(path, "r", encoding="utf-8") as f:
            # 逐行读取，行号从1开始
            for line_no, line in enumerate(f, 1):
                line = line.strip() # 去除首尾空白
                if not line: continue # 空行直接跳过
                # 把每行JSON字符串解析成Python字典
                obj = json.loads(line)
                # 取text字段，兜底空串再去空格
                text = (obj.get("text") or "").strip()
                label = obj.get("label") # 取label字段
                if not text: # text为空时抛错，防止训练空字符串
                    raise ValueError(f"第 {line_no} 行 text 为空")
                if label not in LABEL2ID: # label不在白名单里，抛错
                    raise ValueError(
                        f"第 {line_no} 行 label 非法: {label!r}")
                # 校验通过，加入结果
                rows.append({"text": text, "label": label})
        if not rows: # 整个文件没读到有效数据
            raise ValueError(f"训练数据为空：{path}")
        return rows

    # 切分 训练集/验证集/测试集
    @staticmethod
    def _stratified_split(
        rows: list[dict],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """按标签做分层切分，保证训练/验证/测试集里每类比例一致"""
        random.seed(seed) # 固定随机种子，每次切分结果一样
        # 按标签分桶，key是标签名
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            # 把每条数据按label扔进对应的桶里
            buckets.setdefault(row["label"], []).append(row)
            '''
            1. 取当前行的分类标签 row["label"]
            2. 去 buckets 字典查找这个标签：
            2.1. 找不到：自动新增键值对 label: []，返回空列表 []
            2.2. 找得到：返回该标签已有的数据列表
            3. 拿到列表后执行.append(row)，把当前行数据加入这个标签分组
            '''
        # 三个空列表准备接结果
        train_rows, val_rows, test_rows = [], [], []
        for label, group in buckets.items(): # 逐个桶处理
            random.shuffle(group) # 桶内随机打乱，避免顺序偏差
            n = len(group) # 当前桶里有多少条
            n_test = max(1, int(n * test_ratio)) # 测试集，至少1条
            n_val  = max(1, int(n * val_ratio)) # 验证集，至少1条
            test_rows.extend(group[:n_test]) # 前n_test条分给测试集
            # 接着n_val条分给验证集
            val_rows.extend(group[n_test:n_test + n_val])
            # 剩下的全给训练集
            train_rows.extend(group[n_test + n_val:])

        random.shuffle(train_rows) # 训练集整体再洗一次，防止同标签扎堆
        # 返回训练集、验证集、测试集
        return train_rows, val_rows, test_rows


# ── 模块级单例（供外部 import 调用，确保全局只加载一次模型）────
_classifier: Optional[QueryClassifier] = None # 全局单例缓存

def get_query_classifier() -> QueryClassifier:
    """获取 QueryClassifier 单例（懒加载，首次调用时初始化）"""
    global _classifier
    if _classifier is None: # 首次调用时创建单例
        _classifier = QueryClassifier.get_instance()
    return _classifier # 返回单例


if __name__ == '__main__':
    # 加载项目根目录的 .env.local 环境变量
    from dotenv import load_dotenv
    load_dotenv("../../.env.local")

    # ===============微调训练模型========================
    # 初始化训练模式（加载基座模型）
    qc = QueryClassifier(train_model=True)
    # 微调训练
    qc.train(
        data_path="../../tmp/training_data.jsonl", # 训练数据文件
        epochs=8, # 训练轮数
        batch_size=64, # 训练批次大小
    )
    # 输出：✅ 微调完成，模型保存到配置的微调模型路径

    input('➡️ 开始测试微调后的模型：')
    # =============测试微调后的模型==============================
    qc = QueryClassifier() # 自动加载微调后的模型做推理

    test_cases = [
        # 通用概念
        ("什么是 Java 多态？", "general"),
        ("HTTP 和 HTTPS 的区别？", "general"),
        # 课程专属
        ("EduAgent 的 Reranker 置信度阈值是多少？", "specialized"),
        ("第3章实战项目怎么配置 Spring 数据源？", "specialized"),
        # 故意写错，测试一个，这就应该是 通用概念general
        ("BGE-M3 和 BGE-Reranker 有什么区别？", "specialized"),
    ]

    # 打印表头
    print(f"{'Query':<40} {'预期':<12} {'预测':<12} {'置信度'}")
    print("-" * 75)
    for text, expected in test_cases:
        label, conf = qc.classify(text) # 调用分类器
        mark = "✓" if label == expected else "✗" # 对比预期结果
        print(f"{text:<40} {expected:<12} "
              f"{label:<12} {conf:.4f}  {mark}") # 输出分类结果
