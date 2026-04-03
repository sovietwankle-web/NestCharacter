# -*- coding: utf-8 -*-
"""
FastAPI 服务 - 嵌套字符识别 API
用于集成到 Coze 平台
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import cv2
import numpy as np
import os
import tempfile
import base64
from datetime import datetime

from nested_char_detector import NestedCharDetector

# 创建 FastAPI 应用
app = FastAPI(
    title="嵌套字符识别 API",
    description="基于深度学习和分形理论的嵌套字符识别系统",
    version="1.0.0"
)

# 配置 CORS（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化检测器（使用已训练的模型）
MODEL_PATH = "models/nested_char_model.pth"
detector = None

# 临时文件目录
TEMP_DIR = tempfile.gettempdir()


class DetectionResponse(BaseModel):
    """检测结果响应模型"""
    success: bool
    message: str
    is_nested: bool
    estimated_layers: int
    confidence: float
    transform_params: Optional[Dict]
    pyramid_analysis: Optional[Dict]
    num_boxes: int
    ocr_boxes: Optional[List]
    processing_time: float


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model_loaded: bool
    timestamp: str


class APIInfoResponse(BaseModel):
    """API 信息响应模型"""
    name: str
    version: str
    description: str
    endpoints: List[Dict]


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    global detector

    # 检查模型文件是否存在
    if os.path.exists(MODEL_PATH):
        detector = NestedCharDetector(model_path=MODEL_PATH)
        print(f"✅ 模型已加载: {MODEL_PATH}")
    else:
        # 没有模型也可以运行，使用默认初始化
        detector = NestedCharDetector()
        print(f"⚠️  未找到训练好的模型，使用默认初始化")


@app.get("/", response_model=APIInfoResponse)
async def root():
    """API 根路径 - 返回 API 信息"""
    return APIInfoResponse(
        name="嵌套字符识别 API",
        version="1.0.0",
        description="基于深度学习和分形理论的嵌套字符识别系统，支持检测多层嵌套字符结构",
        endpoints=[
            {
                "path": "/api/detect",
                "method": "POST",
                "description": "检测图片中的嵌套字符结构"
            },
            {
                "path": "/api/health",
                "method": "GET",
                "description": "健康检查"
            },
            {
                "path": "/api/info",
                "method": "GET",
                "description": "获取 API 信息"
            }
        ]
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/info", response_model=APIInfoResponse)
async def api_info():
    """获取 API 详细信息"""
    return APIInfoResponse(
        name="嵌套字符识别 API",
        version="1.0.0",
        description="支持检测0-5层的嵌套字符结构，提供变换恢复和OCR框生成功能",
        endpoints=[
            {
                "path": "/api/detect",
                "method": "POST",
                "description": "上传图片进行嵌套字符检测",
                "parameters": {
                    "file": "图片文件 (支持 PNG, JPG, JPEG)"
                },
                "returns": {
                    "is_nested": "是否为嵌套字符",
                    "estimated_layers": "估计的嵌套层数 (0-5)",
                    "confidence": "置信度 (0-1)",
                    "num_boxes": "检测到的OCR框数量",
                    "transform_params": "检测到的图像变换参数",
                    "pyramid_analysis": "高斯金字塔分析结果"
                }
            },
            {
                "path": "/api/health",
                "method": "GET",
                "description": "检查服务健康状态"
            }
        ]
    )


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_nested_char(file: UploadFile = File(...)):
    """
    检测嵌套字符

    参数:
        file: 上传的图片文件

    返回:
        检测结果，包括是否为嵌套字、层数、置信度等信息
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="检测器未初始化")

    # 检查文件类型
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。请上传以下格式之一: {', '.join(allowed_extensions)}"
        )

    # 保存临时文件
    temp_file_path = os.path.join(TEMP_DIR, f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{file_ext}")

    try:
        # 读取上传的文件
        contents = await file.read()

        # 保存到临时文件
        with open(temp_file_path, 'wb') as f:
            f.write(contents)

        # 记录开始时间
        start_time = datetime.now()

        # 执行检测
        result = detector.detect(temp_file_path)

        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()

        # 构建响应
        response = DetectionResponse(
            success=True,
            message="检测成功",
            is_nested=result['is_nested'],
            estimated_layers=result['estimated_layers'],
            confidence=result['confidence'],
            transform_params=result.get('transform_params'),
            pyramid_analysis={
                'edge_densities': result['pyramid_analysis']['edge_densities'],
                'char_sizes': result['pyramid_analysis']['char_sizes']
            },
            num_boxes=result['num_boxes'],
            ocr_boxes=result['ocr_boxes'],
            processing_time=processing_time
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@app.post("/api/detect_base64")
async def detect_base64(data: Dict):
    """
    检测嵌套字符 (Base64 输入)

    参数:
        data: {"image": "base64编码的图片数据"}

    返回:
        检测结果
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="检测器未初始化")

    if "image" not in data:
        raise HTTPException(status_code=400, detail="缺少 'image' 字段")

    try:
        # 解码 base64 图片
        image_data = base64.b64decode(data["image"])

        # 转换为 numpy 数组
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图片数据")

        # 保存临时文件
        temp_file_path = os.path.join(TEMP_DIR, f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
        cv2.imwrite(temp_file_path, image)

        # 记录开始时间
        start_time = datetime.now()

        # 执行检测
        result = detector.detect(temp_file_path)

        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()

        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # 构建响应
        response = {
            "success": True,
            "message": "检测成功",
            "is_nested": result['is_nested'],
            "estimated_layers": result['estimated_layers'],
            "confidence": result['confidence'],
            "num_boxes": result['num_boxes'],
            "processing_time": processing_time
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


if __name__ == "__main__":
    # 运行服务器
    print("🚀 启动嵌套字符识别 API 服务...")
    print("📝 API 文档: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/api/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
