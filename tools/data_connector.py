"""
真实数据连接器
"""
import pandas as pd
import requests
import os
from modules.YA_Common.utils.logger import get_logger
from . import YA_MCPServer_Tool

logger = get_logger("DataConnector")

@YA_MCPServer_Tool(
    name="download_retail_dataset",
    title="下载零售数据集",
    description="下载UCI在线零售数据集"
)
def download_retail_dataset() -> str:
    """下载UCI在线零售数据集"""
    
    logger.info("开始下载UCI在线零售数据集...")
    
    # 数据集URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    try:
        # 创建resources目录
        os.makedirs('resources', exist_ok=True)
        
        # 下载文件
        logger.info("正在下载数据文件...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # 保存文件
            with open('resources/Online_Retail.xlsx', 'wb') as f:
                f.write(response.content)
            
            logger.info("数据文件下载完成")
            return "✅ 数据集下载成功：resources/Online_Retail.xlsx"
        else:
            return f"❌ 下载失败，HTTP状态码：{response.status_code}"
            
    except Exception as e:
        logger.error(f"下载失败：{str(e)}")
        return f"❌ 下载失败：{str(e)}"

@YA_MCPServer_Tool(
    name="load_and_preview_data",
    title="加载并预览数据",
    description="加载下载的数据集并显示基本信息"
)
def load_and_preview_data() -> dict:
    """加载并预览数据"""
    
    try:
        logger.info("正在加载数据集...")
        
        # 读取Excel文件
        df = pd.read_excel('resources/Online_Retail.xlsx')
        
        # 基本信息
        info = {
            "数据形状": f"{df.shape[0]} 行 x {df.shape[1]} 列",
            "列名": df.columns.tolist(),
            "缺失值": df.isnull().sum().to_dict(),
            "数据类型": df.dtypes.astype(str).to_dict(),
            "前5行数据": df.head().to_dict('records')
        }
        
        logger.info(f"数据加载成功：{df.shape[0]} 行 x {df.shape[1]} 列")
        
        return info
        
    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}")
        return {"error": f"数据加载失败：{str(e)}"}
