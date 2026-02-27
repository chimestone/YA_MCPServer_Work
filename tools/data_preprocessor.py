# tools/data_preprocessor.py
from . import YA_MCPServer_Tool
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

@YA_MCPServer_Tool(
    name="data_preprocess",
    title="数据预处理",
    description="对输入数据进行清洗、编码、归一化等预处理"
)
def data_preprocess(
    data: str,
    task_type: str = "表格",
    missing_strategy: str = "均值填充",
    normalize: bool = True
) -> Dict[str, Any]:
    """
    数据预处理
    
    Args:
        data: JSON格式的数据或CSV路径
        task_type: 任务类型(文本/图像/表格)
        missing_strategy: 缺失值处理策略(删除/均值填充/中位数填充/众数填充)
        normalize: 是否归一化
    """
    try:
        # 解析数据
        if data.endswith('.csv'):
            df = pd.read_csv(data)
        else:
            df = pd.DataFrame(json.loads(data))
        
        original_shape = df.shape
        processing_log = []
        
        # 缺失值处理
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            if missing_strategy == "删除":
                df = df.dropna()
                processing_log.append(f"删除了 {missing_count} 个缺失值")
            elif missing_strategy == "均值填充":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                processing_log.append(f"使用均值填充了 {missing_count} 个缺失值")
            elif missing_strategy == "中位数填充":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                processing_log.append(f"使用中位数填充了 {missing_count} 个缺失值")
            elif missing_strategy == "众数填充":
                df = df.fillna(df.mode().iloc[0])
                processing_log.append(f"使用众数填充了 {missing_count} 个缺失值")
        
        # 类别编码
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            processing_log.append(f"对 {len(categorical_cols)} 个类别特征进行了标签编码")
        
        # 归一化
        if normalize:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                processing_log.append(f"对 {len(numeric_cols)} 个数值特征进行了标准化")
        
        return {
            "status": "success",
            "original_shape": original_shape,
            "processed_shape": df.shape,
            "processing_log": processing_log,
            "data_preview": df.head(5).to_dict(),
            "statistics": {
                "total_features": len(df.columns),
                "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_features": len(categorical_cols),
                "total_samples": len(df)
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "数据预处理失败，请检查输入格式"
        }
