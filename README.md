# 灵犀-FastAPI

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

将 中国移动 网页端模型灵犀封装为兼容 OpenAI API 的 API Server。受[Nativu5/Gemini-FastAPI](https://github.com/Nativu5/Gemini-FastAPI/) 启发，代码由Claude协助完成。

**✅ 无需 API Key，免费通过 API 调用 通义Qwen-3.7-Plus、Deekseek-4.0-Pro、豆包Seed 2.0-lite（文生图）等模型！**

## 功能特性

- 🔐 **无需API Key**：只需网页 Cookie，即可免费通过 API 调用 灵犀提供的通义Qwen-3.7-Plus、Deekseek-4.0-Pro、豆包Seed 2.0-lite（文生图）等模型。
- 🔍 **灵活的网络搜索**：搜索功能可通过网页/APP随时开关，模型响应更加准确。
- 🖼️ **多模型支持**：在网页/APP随时切换大模型，通义Qwen-3.7-Plus、Deekseek-4.0-Pro等随时响应。
- ⚖️ **负载均衡**：支持多账户轮询，可同时使用多个账户保证文生图的token消耗。
- 💰 **免费无限量**：使用文字对话无限量无费用；文生图等需要消耗token，可通过中国移动APP领取及购买

## 快速开始

### 前置条件

- Python 3.10
- 拥有网页版 中国移动邮箱 访问权限的账号
- 从 中国移动邮箱 网页获取的 `Authorization` 和 `user_id` 的Cookie

### 安装

```bash
pip install -r requirements.txt
copy config\config.example.yaml config\config.yaml
```

### 配置

编辑 `config/config.yaml` 并提供至少一组凭证：

```yaml
yun139:
  clients:
    - id: "client-a"
      authorization: "Basic ..."
      user_id: "10...."
      source_channel: "101"
```

### 启动服务

```bash
# 直接用 Python
python run.py
```

服务默认启动在 `http://localhost:3000`。

## API 接口

本服务器支持 OpenAI 兼容协议。

### OpenAI 兼容接口

这些接口遵循 OpenAI 的 API 规范，允许你将灵犀直接接入现有的 AI 应用。

- **`GET /v1/models`**: 列出所有可用的 Gemini 模型。
- **`POST /v1/chat/completions`**: 统一聊天对话接口。
  - **流式传输**: 设置 `stream: true` 即可实时接收增量响应 (Stream Delta)。
  - **多模态支持**: 支持在消息中包含文本、图片以及文件上传。
  - **工具调用**: 支持通过 `tools` 参数进行函数调用 (Function Calling)。
  - **结构化输出**: 支持 `response_format`，可严格遵循 JSON Schema。

## 免责声明

本项目与 中国移动 或 OpenAI 无关，仅供学习和研究使用。本项目使用了逆向工程 API，可能不符合 中国移动 服务条款。使用风险自负。
