# resources/model_database.py
from . import YA_MCPServer_Resource

@YA_MCPServer_Resource(
    uri="model://database",
    title="模型数据库",
    description="提供分类算法模型的元数据与性能基准",
    mime_type="application/json"
)
def model_database() -> str:
    """返回模型数据库信息"""
    return """{
  "models": [
    {
      "name": "逻辑回归",
      "type": "线性模型",
      "input_format": "数值特征向量",
      "output_format": "类别标签 + 概率",
      "compute_requirement": "低",
      "memory_usage": "10-50MB",
      "training_time": "秒级",
      "inference_time": "毫秒级",
      "benchmark": {
        "accuracy": 0.82,
        "precision": 0.81,
        "recall": 0.83,
        "f1_score": 0.82
      }
    },
    {
      "name": "SVM",
      "type": "支持向量机",
      "input_format": "数值特征向量",
      "output_format": "类别标签 + 概率",
      "compute_requirement": "中",
      "memory_usage": "50-200MB",
      "training_time": "分钟级",
      "inference_time": "毫秒级",
      "benchmark": {
        "accuracy": 0.88,
        "precision": 0.87,
        "recall": 0.89,
        "f1_score": 0.88
      }
    },
    {
      "name": "朴素贝叶斯",
      "type": "概率模型",
      "input_format": "数值特征向量",
      "output_format": "类别标签 + 概率",
      "compute_requirement": "低",
      "memory_usage": "10-30MB",
      "training_time": "秒级",
      "inference_time": "毫秒级",
      "benchmark": {
        "accuracy": 0.80,
        "precision": 0.79,
        "recall": 0.81,
        "f1_score": 0.80
      }
    },
    {
      "name": "随机森林",
      "type": "集成学习",
      "input_format": "数值/类别特征",
      "output_format": "类别标签 + 概率",
      "compute_requirement": "中",
      "memory_usage": "100-500MB",
      "training_time": "分钟级",
      "inference_time": "毫秒级",
      "benchmark": {
        "accuracy": 0.90,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90
      }
    },
    {
      "name": "XGBoost",
      "type": "梯度提升",
      "input_format": "数值/类别特征",
      "output_format": "类别标签 + 概率",
      "compute_requirement": "高",
      "memory_usage": "200-1GB",
      "training_time": "分钟-小时级",
      "inference_time": "毫秒级",
      "benchmark": {
        "accuracy": 0.93,
        "precision": 0.92,
        "recall": 0.94,
        "f1_score": 0.93
      }
    }
  ]
}"""
