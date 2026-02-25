# tools/product_classifier_enhanced.py
from . import YA_MCPServer_Tool
from .product_classifier_models import ProductClassifierEnsemble
from .dataset_loader import prepare_training_data_from_dataset, TaobaoDatasetLoader
from typing import List, Dict, Any, Optional
import os
import pandas as pd
from modules.YA_Common.utils.logger import get_logger

logger = get_logger("ProductClassifierEnhanced")

# 全局模型实例
_classifier = ProductClassifierEnsemble()
_model_path = "models/product_classifier_enhanced.pkl"
_dataset_loader = TaobaoDatasetLoader()

def _ensure_model_trained():
    """确保模型已训练"""
    global _classifier
    
    # 尝试加载已保存的模型
    if os.path.exists(_model_path):
        if _classifier.load_model(_model_path):
            return
    
    # 如果没有保存的模型，则使用真实数据训练新模型
    logger.info("开始使用真实数据集训练商品分类模型...")
    texts, categories, subcategories = prepare_training_data_from_dataset()
    
    # 创建模型目录
    os.makedirs(os.path.dirname(_model_path), exist_ok=True)
    
    # 训练模型
    training_results = _classifier.train(texts, categories)
    
    # 保存模型
    _classifier.save_model(_model_path)
    
    logger.info("真实数据集模型训练并保存完成")
    logger.info(f"训练结果: {training_results}")

@YA_MCPServer_Tool(
    name="classify_product_with_real_data",
    title="商品分类（真实数据训练）",
    description="使用真实电商数据集训练的多算法集成模型进行商品分类"
)
def classify_product_with_real_data(
    product_title: str,
    algorithms: Optional[List[str]] = None,
    return_confidence_analysis: bool = True
) -> Dict[str, Any]:
    """
    使用真实数据训练的模型进行商品分类
    
    Args:
        product_title: 商品标题
        algorithms: 使用的算法列表
        return_confidence_analysis: 是否返回置信度分析
    
    Returns:
        详细的分类结果
    """
    try:
        _ensure_model_trained()
        
        if algorithms is None:
            algorithms = ['ensemble', 'svm', 'naive_bayes', 'random_forest']
        
        # 进行预测
        results = _classifier.predict(product_title, algorithms)
        
        # 构建详细响应
        response = {
            "input": {
                "product_title": product_title,
                "algorithms_requested": algorithms
            },
            "prediction": {},
            "analysis": {},
            "metadata": {
                "timestamp": str(pd.Timestamp.now()),
                "model_version": "enhanced_v1.0"
            }
        }
        
        # 主要预测结果
        if 'ensemble' in results:
            response["prediction"] = {
                "category": results['ensemble']['prediction'],
                "confidence": round(results['ensemble']['confidence'], 4),
                "confidence_level": _get_confidence_level(results['ensemble']['confidence']),
                "method": "ensemble_voting"
            }
        
        # 算法详细结果
        if len(results) > 1:
            response["analysis"]["individual_algorithms"] = {}
            for algo, result in results.items():
                response["analysis"]["individual_algorithms"][algo] = {
                    "prediction": result['prediction'],
                    "confidence": round(result['confidence'], 4),
                    "confidence_level": _get_confidence_level(result['confidence'])
                }
        
        # 置信度分析
        if return_confidence_analysis and len(results) > 1:
            confidences = [r['confidence'] for r in results.values()]
            predictions = [r['prediction'] for r in results.values()]
            
            response["analysis"]["confidence_analysis"] = {
                "average_confidence": round(sum(confidences) / len(confidences), 4),
                "max_confidence": round(max(confidences), 4),
                "min_confidence": round(min(confidences), 4),
                "prediction_consensus": len(set(predictions)) == 1,
                "consensus_rate": predictions.count(predictions[0]) / len(predictions)
            }
        
        # 添加相似商品建议（基于训练数据）
        similar_products = _find_similar_products(product_title)
        if similar_products:
            response["analysis"]["similar_products_in_training"] = similar_products
        
        logger.info(f"商品分类完成: {product_title} -> {response['prediction'].get('category', 'Unknown')}")
        
        return response
        
    except Exception as e:
        logger.error(f"商品分类失败: {str(e)}")
        return {
            "error": f"分类失败: {str(e)}",
            "input": {"product_title": product_title}
        }

@YA_MCPServer_Tool(
    name="get_dataset_statistics",
    title="获取数据集统计信息",
    description="获取训练数据集的详细统计信息"
)
def get_dataset_statistics() -> Dict[str, Any]:
    """获取数据集统计信息"""
    try:
        stats = _dataset_loader.get_statistics()
        
        # 添加更详细的分析
        enhanced_stats = {
            "dataset_overview": {
                "total_products": stats['total_products'],
                "unique_categories": stats['category_count'],
                "unique_subcategories": stats['subcategory_count']
            },
            "category_distribution": stats['categories'],
            "subcategory_distribution": stats['subcategories'],
            "data_quality": {
                "average_title_length": _get_average_title_length(),
                "category_balance": _analyze_category_balance(stats['categories'])
            },
            "model_info": {
                "algorithms_available": ["SVM", "朴素贝叶斯", "随机森林", "集成投票"],
                "feature_extraction": "TF-IDF + 中文分词",
                "training_status": "已训练" if _classifier.is_trained else "未训练"
            }
        }
        
        return enhanced_stats
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return {"error": f"获取统计信息失败: {str(e)}"}

@YA_MCPServer_Tool(
    name="evaluate_model_performance",
    title="模型性能评估",
    description="评估各个算法在测试集上的性能表现"
)
def evaluate_model_performance() -> Dict[str, Any]:
    """评估模型性能"""
    try:
        _ensure_model_trained()
        
        # 重新加载数据进行评估
        texts, categories, _ = prepare_training_data_from_dataset()
        
        # 简单的性能评估（在实际项目中应该使用独立的测试集）
        sample_size = min(50, len(texts))  # 取样本进行快速评估
        sample_indices = np.random.choice(len(texts), sample_size, replace=False)
        
        correct_predictions = {"svm": 0, "naive_bayes": 0, "random_forest": 0, "ensemble": 0}
        total_predictions = sample_size
        
        for i in sample_indices:
            text = texts[i]
            true_category = categories[i]
            
            predictions = _classifier.predict(text, ['svm', 'naive_bayes', 'random_forest', 'ensemble'])
            
            for algo, result in predictions.items():
                if result['prediction'] == true_category:
                    correct_predictions[algo] += 1
        
        # 计算准确率
        accuracies = {algo: correct / total_predictions for algo, correct in correct_predictions.items()}
        
        return {
            "evaluation_summary": {
                "sample_size": sample_size,
                "total_training_data": len(texts)
            },
            "algorithm_performance": {
                algo: {
                    "accuracy": round(acc, 4),
                    "correct_predictions": correct_predictions[algo],
                    "total_predictions": total_predictions
                }
                for algo, acc in accuracies.items()
            },
            "best_algorithm": max(accuracies.items(), key=lambda x: x[1])[0],
            "performance_ranking": sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"性能评估失败: {str(e)}")
        return {"error": f"性能评估失败: {str(e)}"}

def _find_similar_products(query_title: str, top_k: int = 3) -> List[Dict[str, str]]:
    """在训练数据中找到相似商品"""
    try:
        df = _dataset_loader.load_dataset()
        
        # 简单的相似度计算（基于关键词重叠）
        query_words = set(query_title.lower().split())
        similarities = []
        
        for _, row in df.iterrows():
            title_words = set(row['title'].lower().split())
            similarity = len(query_words & title_words) / len(query_words | title_words)
            similarities.append((similarity, row['title'], row['category']))
        
        # 排序并返回top_k
        similarities.sort(reverse=True)
        
        return [
            {
                "title": title,
                "category": category,
                "similarity": round(sim, 3)
            }
            for sim, title, category in similarities[:top_k]
            if sim > 0.1  # 只返回有一定相似度的
        ]
        
    except Exception as e:
        logger.error(f"查找相似商品失败: {str(e)}")
        return []

def _get_average_title_length() -> float:
    """计算平均标题长度"""
    try:
        df = _dataset_loader.load_dataset()
        return round(df['title'].str.len().mean(), 2)
    except:
        return 0.0

def _analyze_category_balance(categories: Dict[str, int]) -> str:
    """分析类别平衡性"""
    if not categories:
        return "无数据"
    
    values = list(categories.values())
    max_count = max(values)
    min_count = min(values)
    
    if max_count / min_count <= 2:
        return "平衡"
    elif max_count / min_count <= 5:
        return "轻微不平衡"
    else:
        return "严重不平衡"

def _get_confidence_level(confidence: float) -> str:
    """根据置信度返回等级"""
    if confidence >= 0.8:
        return "高"
    elif confidence >= 0.6:
        return "中"
    else:
        return "低"

# 导入numpy用于随机采样
import numpy as np
