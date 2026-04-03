# -*- coding: utf-8 -*-
"""
API 测试脚本
用于测试嵌套字符识别 API
"""

import requests
import json
import sys
import os
from pathlib import Path


def test_health_check(base_url: str):
    """测试健康检查端点"""
    print("\n" + "="*50)
    print("测试 1: 健康检查")
    print("="*50)

    url = f"{base_url}/api/health"
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        print(f"响应内容:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_api_info(base_url: str):
    """测试 API 信息端点"""
    print("\n" + "="*50)
    print("测试 2: API 信息")
    print("="*50)

    url = f"{base_url}/api/info"
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        print(f"响应内容:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_detect(base_url: str, image_path: str):
    """测试检测端点"""
    print("\n" + "="*50)
    print("测试 3: 嵌套字符检测")
    print("="*50)

    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return False

    url = f"{base_url}/api/detect"
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            response = requests.post(url, files=files)

        print(f"状态码: {response.status_code}")
        print(f"响应内容:")

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 显示关键信息
            print("\n" + "-"*50)
            print("📊 检测结果摘要:")
            print("-"*50)
            print(f"是否为嵌套字: {'✅ 是' if result.get('is_nested') else '❌ 否'}")
            print(f"估计层数: {result.get('estimated_layers')}")
            print(f"置信度: {result.get('confidence', 0):.2%}")
            print(f"检测框数量: {result.get('num_boxes')}")
            print(f"处理时间: {result.get('processing_time', 0):.3f} 秒")

            transform = result.get('transform_params', {})
            if transform:
                print(f"\n🔄 检测到的变换:")
                print(f"  旋转角度: {transform.get('rotation', 0):.2f}°")
                print(f"  X轴缩放: {transform.get('scale_x', 1.0):.3f}")
                print(f"  Y轴缩放: {transform.get('scale_y', 1.0):.3f}")
                print(f"  水平翻转: {'是' if transform.get('flip_horizontal') else '否'}")
                print(f"  垂直翻转: {'是' if transform.get('flip_vertical') else '否'}")

            return True
        else:
            print(f"❌ 错误响应: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    """主函数"""
    # API 基础 URL
    base_url = "http://localhost:8000"

    # 如果提供了命令行参数，使用自定义 URL
    if len(sys.argv) > 1:
        base_url = sys.argv[1]

    print("🚀 开始测试嵌套字符识别 API")
    print(f"📡 API 地址: {base_url}")

    # 测试健康检查
    health_ok = test_health_check(base_url)

    # 测试 API 信息
    info_ok = test_api_info(base_url)

    # 测试检测功能（需要提供测试图片）
    detect_ok = False

    # 查找测试图片
    test_images = []

    # 检查常见的图片目录
    image_dirs = ['input_images', 'output_images', 'training_data', '.']

    for dir_name in image_dirs:
        if os.path.exists(dir_name):
            for ext in ['.png', '.jpg', '.jpeg']:
                test_images.extend(list(Path(dir_name).glob(f'*{ext}')))

    if test_images:
        # 使用第一张图片测试
        test_image = str(test_images[0])
        print(f"\n📸 使用测试图片: {test_image}")
        detect_ok = test_detect(base_url, test_image)
    else:
        print("\n⚠️  未找到测试图片，跳过检测测试")
        print("   你可以手动指定图片路径:")
        print(f"   python test_api.py {base_url} <image_path>")

        if len(sys.argv) > 2:
            test_image = sys.argv[2]
            detect_ok = test_detect(base_url, test_image)

    # 总结
    print("\n" + "="*50)
    print("📋 测试总结")
    print("="*50)
    print(f"健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"API 信息: {'✅ 通过' if info_ok else '❌ 失败'}")
    print(f"嵌套检测: {'✅ 通过' if detect_ok else '⚠️  跳过/失败'}")

    if health_ok and info_ok:
        print("\n✅ API 服务运行正常！")
        print(f"\n📝 查看完整 API 文档: {base_url}/docs")
        print(f"🔍 Swagger UI: {base_url}/docs")
        print(f"📖 ReDoc: {base_url}/redoc")
        return 0
    else:
        print("\n❌ API 服务存在问题，请检查服务是否正常启动")
        return 1


if __name__ == "__main__":
    sys.exit(main())
