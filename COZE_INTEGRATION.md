# Coze 平台集成指南

本文档介绍如何将嵌套字符识别系统集成到 Coze（扣子）平台。

## 📋 前置准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件（可选）

如果有训练好的模型，请放置在 `models/nested_char_model.pth`

没有模型也可以运行，系统会使用默认初始化。

## 🚀 启动 API 服务

### 本地测试

```bash
python api_server.py
```

服务将在 `http://localhost:8000` 启动

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/health
- API 信息: http://localhost:8000/api/info

### 测试 API

使用 curl 测试：

```bash
# 健康检查
curl http://localhost:8000/api/health

# 检测图片
curl -X POST "http://localhost:8000/api/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.png"
```

## 🌐 部署到公网

Coze 需要访问公网 API，你可以选择以下部署方式：

### 方案 1: 使用 ngrok（测试推荐）

```bash
# 安装 ngrok
# 下载地址: https://ngrok.com/download

# 启动 API 服务
python api_server.py

# 在另一个终端启动 ngrok
ngrok http 8000
```

ngrok 会生成一个公网 URL，例如: `https://xxxx-xx-xx-xx-xx.ngrok-free.app`

### 方案 2: 部署到云服务器

推荐平台：
- 阿里云
- 腾讯云
- 华为云
- AWS

部署步骤：
1. 购买云服务器（最低配置：2核4G）
2. 安装 Python 和依赖
3. 配置防火墙开放 8000 端口
4. 使用 systemd 或 supervisor 管理服务

### 方案 3: Docker 部署

创建 `Dockerfile`（已包含在项目中）:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
```

构建和运行：

```bash
docker build -t nested-char-api .
docker run -p 8000:8000 nested-char-api
```

## 🔧 在 Coze 平台创建插件

### 步骤 1: 登录 Coze 平台

访问 https://www.coze.cn/ 并登录

### 步骤 2: 创建工作空间

1. 点击「创建工作空间」
2. 选择「个人空间」或「团队空间」

### 步骤 3: 创建自定义插件

1. 进入工作空间后，点击「插件」标签
2. 点击「创建插件」→「自定义插件」
3. 选择「API 插件」

### 步骤 4: 配置插件信息

#### 基本信息

- **插件名称**: 嵌套字符识别
- **插件描述**: 基于深度学习和分形理论的嵌套字符识别系统，可以检测图片中的多层嵌套字符结构
- **插件图标**: 上传一个图标（可选）

#### API 配置

##### 方式一：使用 OpenAPI/Swagger 导入（推荐）

1. 访问你的 API 文档地址: `http://your-domain/docs`
2. 点击「GET /openapi.json」获取 OpenAPI 规范
3. 复制 JSON 内容
4. 在 Coze 平台选择「导入 OpenAPI」
5. 粘贴 JSON 内容并保存

##### 方式二：手动配置

**1. 配置服务器地址**

```
基础 URL: http://your-domain  (替换为你的实际域名或 ngrok URL)
```

**2. 添加 API 端点**

##### 端点 1: 健康检查

- **路径**: `/api/health`
- **方法**: `GET`
- **描述**: 检查服务健康状态
- **返回示例**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-14T10:00:00"
}
```

##### 端点 2: 检测嵌套字符（主要功能）

- **路径**: `/api/detect`
- **方法**: `POST`
- **描述**: 上传图片检测嵌套字符结构
- **请求类型**: `multipart/form-data`
- **请求参数**:

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| file   | File | 是   | 图片文件（支持 PNG、JPG、JPEG） |

- **返回示例**:
```json
{
  "success": true,
  "message": "检测成功",
  "is_nested": true,
  "estimated_layers": 3,
  "confidence": 0.892,
  "transform_params": {
    "rotation": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "flip_horizontal": false,
    "flip_vertical": false
  },
  "pyramid_analysis": {
    "edge_densities": [0.12, 0.08, 0.05, 0.03, 0.01],
    "char_sizes": [45.2, 32.1, 18.5, 9.2, 4.1]
  },
  "num_boxes": 15,
  "ocr_boxes": [[10, 20, 50, 60], ...],
  "processing_time": 1.23
}
```

**3. 配置授权方式**

目前 API 无需授权，选择「无授权」即可。

如需添加授权，可以修改 `api_server.py` 添加 API Key 验证：

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
```

### 步骤 5: 测试插件

1. 在 Coze 平台的插件配置页面，点击「测试」
2. 上传一张测试图片
3. 查看返回结果

### 步骤 6: 创建 Bot

1. 返回工作空间，点击「创建 Bot」
2. 配置 Bot 基本信息
3. 在「能力」中添加刚才创建的插件
4. 配置提示词，例如：

```
你是一个嵌套字符识别助手。当用户上传图片时，使用嵌套字符识别插件分析图片，并用友好的方式告诉用户：

1. 是否包含嵌套字符
2. 嵌套层数
3. 检测置信度
4. 是否检测到图像变换（旋转、缩放等）

如果检测到嵌套字符，提醒用户这可能是用于规避 OCR 的技术。
```

5. 保存并发布 Bot

## 📝 完整配置示例

### Coze 插件配置 JSON（参考）

```json
{
  "name": "嵌套字符识别",
  "description": "检测图片中的多层嵌套字符结构",
  "baseUrl": "http://your-domain",
  "endpoints": [
    {
      "path": "/api/detect",
      "method": "POST",
      "description": "检测嵌套字符",
      "parameters": [
        {
          "name": "file",
          "in": "formData",
          "required": true,
          "type": "file",
          "description": "要检测的图片文件"
        }
      ],
      "responses": {
        "200": {
          "description": "检测成功",
          "schema": {
            "type": "object",
            "properties": {
              "is_nested": {"type": "boolean"},
              "estimated_layers": {"type": "integer"},
              "confidence": {"type": "number"},
              "num_boxes": {"type": "integer"}
            }
          }
        }
      }
    }
  ]
}
```

## 🧪 使用示例

### 在 Coze Bot 中使用

用户: "帮我检测这张图片是否是嵌套字"
*上传图片*

Bot: "我来帮你分析这张图片..."
*调用嵌套字符识别插件*

Bot: "分析完成！检测结果如下：
- 这是一张嵌套字符图片
- 嵌套层数：3层
- 检测置信度：89.2%
- 检测到 15 个字符区域
- 未检测到图像变换

这种嵌套字符可能是用来规避传统 OCR 识别的技术。"

## ❓ 常见问题

### Q1: API 启动失败

**A**: 检查端口是否被占用，可以修改 `api_server.py` 中的端口号：

```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # 改为 8001
```

### Q2: Coze 无法访问 API

**A**: 确保：
1. API 服务运行在公网可访问的地址
2. 防火墙已开放对应端口
3. 使用 `http://` 或 `https://` 完整 URL

### Q3: 检测速度慢

**A**: 优化建议：
1. 使用 GPU 加速（安装 CUDA 版本的 PyTorch）
2. 减小输入图片尺寸
3. 部署到性能更好的服务器

### Q4: 没有训练好的模型怎么办？

**A**: 可以运行训练脚本生成模型：

```bash
python train_model.py --generate-data --samples-per-class 500 --num-epochs 50
```

## 📚 进阶配置

### 添加 API Key 认证

修改 `api_server.py`：

```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-key-here"

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/api/detect", dependencies=[Depends(verify_api_key)])
async def detect_nested_char(file: UploadFile = File(...)):
    # ... 原有代码
```

在 Coze 配置中添加 Header：
- Header 名称: `X-API-Key`
- Header 值: `your-secret-key-here`

### 添加请求日志

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/api/detect")
async def detect_nested_char(file: UploadFile = File(...)):
    logger.info(f"收到检测请求: {file.filename}")
    # ... 原有代码
```

### 启用 HTTPS

使用反向代理（Nginx）配置 SSL 证书：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🎉 完成！

现在你的嵌套字符识别系统已经成功集成到 Coze 平台，可以通过聊天机器人的方式为用户提供服务了！

如有问题，请查看：
- FastAPI 文档: https://fastapi.tiangolo.com/
- Coze 开发者文档: https://www.coze.cn/open/docs/
