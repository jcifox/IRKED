#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_api.py
国内大模型统一客户端（支持 deepseek/qianwen/ernie/xinghuo/glm）
星火模型使用 OpenAI-兼容 HTTPS 入口，无需签名
"""

# ---------- 基础依赖 ----------
import os
import time
import json
import logging
import requests
from typing import Dict, Optional
from dotenv import load_dotenv

# 一次性加载 .env 文件
load_dotenv()

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------- 客户端实现 ----------
class DomesticLLMClient:
    """
    国内大模型统一客户端
    支持 deepseek / qianwen / ernie / xinghuo / glm
    """

    # 模型 -> 环境变量名 映射
    ENV_MAPPING = {
        "deepseek": "DEEPSEEK_API_KEY",
        "qianwen": "QWEN_API_KEY",
        "ernie": "ERNIE_API_KEY",
        "xinghuo": "SPARK_API_KEY",      # 控制台复制的 APIPassword
        "glm": "ZHIPU_API_KEY",
    }

    # 各模型请求地址
    ENDPOINTS = {
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
        "qianwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "ernie": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
        "xinghuo": "https://spark-api-open.xf-yun.com/v2/chat/completions",  # OpenAI 兼容入口
        "glm": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    }

    # 具体模型代号
    MODEL_CODES = {
        "deepseek": "deepseek-chat",
        "qianwen": "qwen-plus",
        "glm": "glm-4",
        "xinghuo": "spark-x",  
    }

    def __init__(self, model_name: str,
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None):
        """
        初始化客户端
        :param model_name: 模型名称
        :param api_key:  如果为 None，则自动从 .env 读取
        :param secret_key: 部分模型需要，如果为 None，则自动从 .env 读取
        """
        # 转小写并检查支持
        self.model_name = model_name.lower()
        self._check_support()

        # 读取密钥
        self.api_key, _ = self._get_creds(api_key, secret_key)

        # 构造请求地址
        self.endpoint = self._construct_url()

    # ---------- 私有工具 ----------
    def _check_support(self):
        """检查模型是否支持"""
        supported = list(self.ENV_MAPPING.keys())
        if self.model_name not in supported:
            raise ValueError(f"不支持的模型：{self.model_name}，支持：{supported}")

    def _get_creds(self, api_key: Optional[str], _secret_key: Optional[str]) -> tuple[str, None]:
        """
        优先使用外部传入；否则读取 .env
        返回 (api_key, None)  统一格式
        """
        env_var = self.ENV_MAPPING[self.model_name]
        if api_key is None:
            api_key = os.getenv(env_var)
        if api_key is None:
            raise ValueError(f"请在 .env 中配置 {env_var} 或显式传入 api_key")
        return api_key, None

    def _construct_url(self) -> str:
        """
        构造请求地址
        ernie 需要把 access_token 拼在 url
        """
        if self.model_name == "ernie":
            return f"{self.ENDPOINTS['ernie']}?access_token={self.api_key}"
        return self.ENDPOINTS[self.model_name]

    # ---------- 请求/响应 ----------
    def _format_request(self, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """
        统一构造请求体
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        common = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        # 各模型型号
        if self.model_name == "ernie":
            return {**common, "model": "ernie-4.0"}
        return {**common, "model": self.MODEL_CODES.get(self.model_name, self.model_name)}

    def generate(self, prompt: str,
                 system_prompt: Optional[str] = None,
                 retry: int = 2) -> Optional[str]:
        """
        同步生成文本
        :param prompt: 用户提示
        :param system_prompt: 系统提示（可选）
        :param retry: 重试次数
        :return: 模型回复或 None
        """
        data = self._format_request(prompt, system_prompt)

        for i in range(retry):
            try:
                # ----- 构造 headers -----
                headers = {"Content-Type": "application/json"}
                # 除 ernie 外，其余都用 Bearer 令牌
                if self.model_name in ("deepseek", "qianwen", "glm", "xinghuo"):
                    headers["Authorization"] = f"Bearer {self.api_key}"

                # ----- 发送请求 -----
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    data=json.dumps(data, ensure_ascii=False),
                    timeout=2000,
                )
                resp.raise_for_status()

                # ----- 解析 -----
                return self._parse_response(resp.json())

            except Exception as e:
                logger.warning(f"调用失败（{i+1}/{retry}）：{str(e)}")
                if i < retry - 1:
                    time.sleep(1)

        return None

    def _parse_response(self, response: Dict) -> Optional[str]:
        """
        提取回复文本
        """
        try:
            if self.model_name == "ernie":
                return response["result"]
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"解析响应失败：{str(e)}")
            return None


# ---------- 使用示例 ----------
if __name__ == "__main__":
    client = DomesticLLMClient("xinghuo")
    print(client.generate("用一句话介绍星火AI"))