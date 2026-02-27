# tools/classifier_selector.py
from . import YA_MCPServer_Tool
from typing import Dict, Any, Optional, List

@YA_MCPServer_Tool(
    name="classifier_select",
    title="分类算法选择器",
    description="根据用户需求推荐合适的分类算法模型"
)
def classifier_select(
    task_type: str = "文本",
    data_scale: str = "中等",
    accuracy_requirement: str = "高",
    environment: str = "通用"
) -> Dict[str, Any]:
    """
    根据需求推荐分类算法
    
    Args:
        task_type: 任务类型(文本/图像/表格)
        data_scale: 数据规模(小/中等/大)
        accuracy_requirement: 精度要求(低/中/高)
        environment: 运行环境(通用/低资源/高性能)
    """
    recommendations = []
    
    # 逻辑回归
    if data_scale in ["小", "中等"] and accuracy_requirement in ["低", "中"]:
        recommendations.append({
            "model": "逻辑回归",
            "score": 85,
            "pros": ["训练快速", "可解释性强", "内存占用小"],
            "cons": ["处理非线性问题能力弱"],
            "适用场景": "二分类、线性可分数据"
        })
    
    # SVM
    if data_scale in ["小", "中等"] and accuracy_requirement in ["中", "高"]:
        recommendations.append({
            "model": "SVM",
            "score": 90,
            "pros": ["高维数据表现好", "泛化能力强"],
            "cons": ["大数据集训练慢"],
            "适用场景": "文本分类、小样本学习"
        })
    
    # 随机森林
    if accuracy_requirement in ["中", "高"]:
        recommendations.append({
            "model": "随机森林",
            "score": 88,
            "pros": ["抗过拟合", "特征重要性分析", "处理缺失值"],
            "cons": ["模型较大", "预测速度较慢"],
            "适用场景": "表格数据、特征工程"
        })
    
    # XGBoost
    if data_scale in ["中等", "大"] and accuracy_requirement == "高":
        recommendations.append({
            "model": "XGBoost",
            "score": 95,
            "pros": ["准确率高", "支持并行", "内置正则化"],
            "cons": ["参数调优复杂"],
            "适用场景": "竞赛、结构化数据"
        })
    
    # 朴素贝叶斯
    if task_type == "文本" and environment == "低资源":
        recommendations.append({
            "model": "朴素贝叶斯",
            "score": 82,
            "pros": ["极快训练速度", "低内存占用"],
            "cons": ["假设特征独立"],
            "适用场景": "文本分类、垃圾邮件过滤"
        })
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "input": {
            "task_type": task_type,
            "data_scale": data_scale,
            "accuracy_requirement": accuracy_requirement,
            "environment": environment
        },
        "recommendations": recommendations[:3],
        "best_choice": recommendations[0] if recommendations else None,
        "调用建议": "建议使用集成方法结合多个算法以获得最佳效果"
    }
