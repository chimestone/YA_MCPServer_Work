# resources/data_template.py
from . import YA_MCPServer_Resource

@YA_MCPServer_Resource(
    uri="data://template",
    title="数据格式模板",
    description="提供不同任务类型的数据格式模板",
    mime_type="application/json"
)
def data_template() -> str:
    """返回数据模板"""
    return """{
  "文本分类": {
    "format": "CSV/JSON",
    "required_fields": ["text", "label"],
    "optional_fields": ["id", "metadata"],
    "example": {
      "text": "这是一个商品标题示例",
      "label": "手机数码"
    },
    "导入指南": "确保文本字段为UTF-8编码，标签需为字符串类型"
  },
  "图像分类": {
    "format": "文件夹结构",
    "required_fields": ["image_path", "label"],
    "structure": "data/类别名/图片文件.jpg",
    "example": {
      "image_path": "data/手机数码/phone001.jpg",
      "label": "手机数码"
    },
    "导入指南": "图片格式支持jpg/png，建议尺寸统一"
  },
  "表格分类": {
    "format": "CSV/Excel",
    "required_fields": ["features", "label"],
    "example": {
      "feature1": 1.5,
      "feature2": 2.3,
      "feature3": "类别A",
      "label": "正类"
    },
    "导入指南": "数值特征无需预处理，类别特征会自动编码"
  }
}"""

@YA_MCPServer_Resource(
    uri="data://evaluation_guide",
    title="评估指标指南",
    description="提供模型评估指标的定义与使用场景",
    mime_type="application/json"
)
def evaluation_guide() -> str:
    """返回评估指标指南"""
    return """{
  "准确率(Accuracy)": {
    "定义": "正确预测的样本数 / 总样本数",
    "计算公式": "(TP + TN) / (TP + TN + FP + FN)",
    "适用场景": "类别平衡的数据集",
    "解读建议": "准确率>0.9为优秀，0.8-0.9为良好，<0.7需优化"
  },
  "精确率(Precision)": {
    "定义": "预测为正类中真正为正类的比例",
    "计算公式": "TP / (TP + FP)",
    "适用场景": "关注误报率，如垃圾邮件检测",
    "解读建议": "精确率高表示预测为正类的可信度高"
  },
  "召回率(Recall)": {
    "定义": "真正为正类中被正确预测的比例",
    "计算公式": "TP / (TP + FN)",
    "适用场景": "关注漏报率，如疾病诊断",
    "解读建议": "召回率高表示正类样本被找全的程度高"
  },
  "F1分数(F1-Score)": {
    "定义": "精确率和召回率的调和平均",
    "计算公式": "2 * (Precision * Recall) / (Precision + Recall)",
    "适用场景": "需要平衡精确率和召回率",
    "解读建议": "F1>0.85为优秀，综合评估模型性能"
  },
  "AUC": {
    "定义": "ROC曲线下的面积",
    "计算公式": "积分计算",
    "适用场景": "二分类问题，评估模型区分能力",
    "解读建议": "AUC>0.9为优秀，0.8-0.9为良好"
  }
}"""
