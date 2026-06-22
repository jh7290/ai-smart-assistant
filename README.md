# AI 智能助手应用

这是一个完整可运行的 AI 智能助手项目，支持多轮对话、角色设定、本地记忆、多会话管理和多模型接入。

## 功能

- 前后端分离式本地 Web 应用
- 多会话创建、切换、历史记录保存
- SQLite 本地持久化消息、角色设定和记忆
- 支持 OpenAI、DeepSeek、豆包等 OpenAI 兼容接口
- 无 API Key 时使用本地兜底回复，方便演示 UI 和流程
- API Key 只放后端 `.env`，不暴露到前端

## 启动

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 5190
```

打开：

```text
http://localhost:5190
```

## 环境配置

复制 `.env.example` 为 `.env`，按需填写：

```env
PORT=5190
AI_PROVIDER=deepseek
AI_BASE_URL=https://api.deepseek.com
AI_API_KEY=你的密钥
AI_MODEL=deepseek-chat
```

## 技术栈

- Python
- FastAPI
- Uvicorn
- SQLite
- Pydantic
- HTML
- CSS
- JavaScript
- Prompt Engineering
- OpenAI-Compatible API
- DeepSeek API
- dotenv

## 接口

- `GET /api/config`
- `GET /api/conversations`
- `POST /api/conversations`
- `GET /api/conversations/:id/messages`
- `POST /api/chat`
- `PATCH /api/conversations/:id/settings`
