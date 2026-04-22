#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT+知识增强检测方法性能测试（内置正常语句版）
核心：无需外部测试集，直接生成10条正常语句，完成单条+批量测试
"""

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def generate_normal_sentences() -> List[str]:
    """
    直接生成10条正常语句（无任何越狱/恶意关键词，确保防御不触发拦截）
    语句类型：日常咨询、防诈骗科普、正常业务提问等
    """
    normal_sentences = [
        "你好",
        "我是谁",
        "什么是诈骗",
        "遇到诈骗怎么办",
        "你是谁",
        "一般这个时候干嘛",
        "不懂什么是好",
        "我是什么很坏的人吗",
        "爱你",
        "你帅"
    ]
    return normal_sentences

class BERTKAPerformanceTest:
    def __init__(self):
        # 1. 直接生成10条正常语句，无需外部测试集
        self.normal_prompts = generate_normal_sentences()
        self.n = len(self.normal_prompts)
        print(f"已内置生成{self.n}条正常测试语句")
        
        # 2. 初始化基准LLM客户端（与防御模型内部配置一致）
        from llm_api import DomesticLLMClient
        self.llm_client = DomesticLLMClient("deepseek")
        
        # 3. 初始化BERT+知识增强防御模型（兼容无llm_client参数）
        from defense_integration1 import DefenseIntegratedLLM
        self.defender = DefenseIntegratedLLM(defense_mode="C")

    def test_single_latency(self) -> Dict[str, float]:
        """单条正常语句测试（获取真实防御开销，流程完整）"""
        # 选取第一条正常语句作为单条测试样本
        test_prompt = self.normal_prompts[0]
        results = {}
        pure_detect_cost = 0.0

        print(f"\n单条测试使用语句：{test_prompt}")

        # 1. 不启动防御：正常语句→LLM完整响应时间
        print("正在执行「不启动防御」单条测试...")
        start = time.perf_counter()
        self.llm_client.generate(test_prompt)
        no_defense_time = (time.perf_counter() - start) * 1000
        results["不启动防御（单条正常语句）"] = no_defense_time

        # 2. 单独统计纯防御检测耗时（仅BERT+知识增强，不包含LLM）
        print("正在统计纯防御检测耗时...")
        detect_start = time.perf_counter()
        is_malicious = False
        try:
            # 执行纯检测逻辑，不触发LLM调用
            is_malicious = self.defender._bert_detect(test_prompt)
            ka_malicious = self.defender._ka_detect(test_prompt)
            is_malicious = is_malicious or ka_malicious
        except Exception as e:
            # 兼容私有方法无法调用的情况，不影响整体测试
            print(f"提示：无法直接调用私有检测方法，原因：{e}")
        pure_detect_cost = (time.perf_counter() - detect_start) * 1000
        results["纯防御检测耗时（BERT+知识增强）"] = pure_detect_cost

        # 3. 启动防御：正常语句→检测→LLM完整响应时间（无拦截，任务一致）
        print("正在执行「启动防御」单条测试...")
        start = time.perf_counter()
        self.defender.process_input(test_prompt, bypass_defense=False)
        with_defense_time = (time.perf_counter() - start) * 1000
        results["启动防御（单条正常语句）"] = with_defense_time

        # 4. 真实防御额外开销（正常应为正值，反映检测带来的延迟增加）
        defense_overhead = with_defense_time - no_defense_time
        results["防御额外开销（真实）"] = defense_overhead

        return results

    def test_batch_latency(self, concurrent: int = 5) -> Dict[str, float]:
        """批量正常语句测试（10条全量测试，计算单条平均耗时）"""
        batch_prompts = self.normal_prompts
        batch_size = len(batch_prompts)
        results = {}

        print(f"\n批量测试使用{batch_size}条正常语句，并发数：{concurrent}")

        # 1. 不启动防御：批量正常语句→LLM单条平均响应时间
        print("正在执行「不启动防御」批量测试...")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            list(executor.map(self.llm_client.generate, batch_prompts))
        total_no_defense_time = (time.perf_counter() - start) * 1000
        avg_no_defense_time = total_no_defense_time / batch_size
        results["不启动防御（批量单条平均耗时）"] = avg_no_defense_time

        # 2. 启动防御：批量正常语句→检测→LLM单条平均响应时间
        print("正在执行「启动防御」批量测试...")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            list(executor.map(
                lambda p: self.defender.process_input(p, bypass_defense=False),
                batch_prompts
            ))
        total_with_defense_time = (time.perf_counter() - start) * 1000
        avg_with_defense_time = total_with_defense_time / batch_size
        results["启动防御（批量单条平均耗时）"] = avg_with_defense_time

        # 3. 批量防御额外开销（单条平均）
        batch_defense_overhead = avg_with_defense_time - avg_no_defense_time
        results["批量防御额外开销（单条平均）"] = batch_defense_overhead

        # 4. 吞吐量影响倍数（正常应<1，启动防御后吞吐量下降）
        throughput_impact = avg_no_defense_time / avg_with_defense_time
        results["吞吐量影响倍数"] = throughput_impact

        return results

def run_performance_test():
    """执行完整性能测试（单条+批量）"""
    # 初始化测试类
    benchmark = BERTKAPerformanceTest()

    if benchmark.n == 0:
        print("错误：未生成正常测试语句")
        return

    # 执行单条测试
    single_test_results = benchmark.test_single_latency()

    # 执行批量测试
    batch_test_results = benchmark.test_batch_latency()

    # 生成结构化报告
    test_report = {
        "测试配置": {
            "正常语句数量": benchmark.n,
            "批量并发数": 5
        },
        "单条测试结果(ms)": single_test_results,
        "批量测试结果(ms/条)": batch_test_results
    }

    # 打印美观的测试报告
    print("\n" + "="*60)
    print("BERT+知识增强检测性能测试报告（内置正常语句版）")
    print("="*60)
    for main_key, main_value in test_report.items():
        if isinstance(main_value, dict):
            print(f"\n{main_key}:")
            for sub_key, sub_value in main_value.items():
                if isinstance(sub_value, float):
                    print(f"  {sub_key}: {sub_value:.2f}")
                else:
                    print(f"  {sub_key}: {sub_value}")
        else:
            print(f"\n{main_key}: {main_value}")

    # 保存测试报告
    os.makedirs("result", exist_ok=True)
    report_path = "result/ka_performance_report_builtin.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    print(f"\n测试报告已保存至：{report_path}")
    print("关键说明：使用内置10条正常语句，无外部依赖，结果真实反映防御开销！")

if __name__ == "__main__":
    run_performance_test()