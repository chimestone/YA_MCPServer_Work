#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试脚本 - 验证所有功能"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tools.product_classifier_enhanced import (
    classify_product_with_real_data,
    get_dataset_statistics,
    evaluate_model_performance
)
from tools.classifier_selector import classifier_select
from tools.data_preprocessor import data_preprocess

def test_classifier():
    """测试商品分类"""
    print("\n" + "="*60)
    print("测试1: 商品分类")
    print("="*60)
    
    test_products = [
        "小米14 Ultra 16GB+1TB 黑色 骁龙8 Gen3",
        "优衣库 男装 圆领T恤 白色 L码",
        "美的 电饭煲 4L 智能预约"
    ]
    
    for product in test_products:
        print(f"\n商品: {product}")
        result = classify_product_with_real_data(product, algorithms=['ensemble'])
        if 'prediction' in result:
            print(f"分类: {result['prediction']['category']}")
            print(f"置信度: {result['prediction']['confidence']:.2%}")
        else:
            print(f"错误: {result.get('error', '未知错误')}")

def test_dataset_stats():
    """测试数据集统计"""
    print("\n" + "="*60)
    print("测试2: 数据集统计")
    print("="*60)
    
    stats = get_dataset_statistics()
    if 'dataset_overview' in stats:
        print(f"\n总商品数: {stats['dataset_overview']['total_products']}")
        print(f"类别数: {stats['dataset_overview']['unique_categories']}")
        print(f"子类别数: {stats['dataset_overview']['unique_subcategories']}")
        print(f"\n类别分布:")
        for cat, count in list(stats['category_distribution'].items())[:3]:
            print(f"  {cat}: {count}")
    else:
        print(f"错误: {stats.get('error', '未知错误')}")

def test_algorithm_selector():
    """测试算法选择"""
    print("\n" + "="*60)
    print("测试3: 算法推荐")
    print("="*60)
    
    result = classifier_select(
        task_type="文本",
        data_scale="中等",
        accuracy_requirement="高"
    )
    
    print(f"\n最佳推荐: {result['best_choice']['model']}")
    print(f"评分: {result['best_choice']['score']}")
    print(f"优点: {', '.join(result['best_choice']['pros'])}")

def test_data_preprocess():
    """测试数据预处理"""
    print("\n" + "="*60)
    print("测试4: 数据预处理")
    print("="*60)
    
    sample_data = '''[
        {"feature1": 1.5, "feature2": 2.3, "category": "A"},
        {"feature1": 2.1, "feature2": 1.8, "category": "B"},
        {"feature1": null, "feature2": 3.2, "category": "A"}
    ]'''
    
    result = data_preprocess(sample_data, missing_strategy="均值填充")
    
    if result['status'] == 'success':
        print(f"\n原始形状: {result['original_shape']}")
        print(f"处理后形状: {result['processed_shape']}")
        print(f"处理日志: {result['processing_log']}")
    else:
        print(f"错误: {result.get('error', '未知错误')}")

def test_model_evaluation():
    """测试模型评估"""
    print("\n" + "="*60)
    print("测试5: 模型性能评估")
    print("="*60)
    
    result = evaluate_model_performance()
    
    if 'best_algorithm' in result:
        print(f"\n最佳算法: {result['best_algorithm']}")
        print(f"\n算法性能排名:")
        for algo, acc in result['performance_ranking']:
            print(f"  {algo}: {acc:.2%}")
    else:
        print(f"错误: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    print("\n" + "[START] 开始测试 MCP Server 功能...")
    
    try:
        test_classifier()
        test_dataset_stats()
        test_algorithm_selector()
        test_data_preprocess()
        test_model_evaluation()
        
        print("\n" + "="*60)
        print("[SUCCESS] 所有测试完成！")
        print("="*60)
        print("\n提示: 运行 'python server.py' 启动MCP服务")
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
