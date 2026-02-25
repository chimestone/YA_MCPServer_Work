# tools/dataset_loader.py
import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, List
from modules.YA_Common.utils.logger import get_logger

logger = get_logger("DatasetLoader")

class TaobaoDatasetLoader:
    """淘宝商品数据集加载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_sample_dataset(self) -> str:
        """
        下载示例数据集
        这里我创建一个模拟真实淘宝数据的样本
        """
        sample_data = [
            # 手机数码 - 智能手机
            ("Apple iPhone 15 Pro Max 1TB 深空黑色 5G手机 A17 Pro芯片", "手机数码", "智能手机"),
            ("华为Mate60 Pro+ 16GB+1TB 雅川青 麒麟9000S 5G手机", "手机数码", "智能手机"),
            ("小米14 Ultra 徕卡影像 16GB+1TB 钛金属 骁龙8 Gen3", "手机数码", "智能手机"),
            ("OPPO Find X7 Ultra 16GB+512GB 大漠银沙 天玑9300", "手机数码", "智能手机"),
            ("vivo X100 Pro+ 16GB+1TB 钛色 天玑9300 蔡司影像", "手机数码", "智能手机"),
            ("荣耀Magic6 Pro 16GB+1TB 雅黑色 骁龙8 Gen3", "手机数码", "智能手机"),
            ("三星Galaxy S24 Ultra 12GB+1TB 钛金灰 骁龙8 Gen3", "手机数码", "智能手机"),
            ("一加12 16GB+1TB 岩黑色 骁龙8 Gen3 哈苏影像", "手机数码", "智能手机"),
            
            # 手机数码 - 笔记本电脑
            ("联想ThinkPad X1 Carbon Gen11 14英寸 i7-1365U 32GB 1TB", "手机数码", "笔记本电脑"),
            ("戴尔XPS 13 Plus 13.4英寸 i7-1360P 16GB 512GB 4K触屏", "手机数码", "笔记本电脑"),
            ("苹果MacBook Pro 16英寸 M3 Max芯片 36GB 1TB 深空灰色", "手机数码", "笔记本电脑"),
            ("华硕ROG幻16 2024款 i9-14900HX RTX4080 32GB 1TB", "手机数码", "笔记本电脑"),
            ("惠普战66六代 14英寸 i5-1340P 16GB 512GB 商务本", "手机数码", "笔记本电脑"),
            ("微软Surface Laptop 5 13.5英寸 i7-1255U 16GB 512GB", "手机数码", "笔记本电脑"),
            
            # 手机数码 - 耳机音响
            ("索尼WH-1000XM5 头戴式无线蓝牙降噪耳机 黑色", "手机数码", "耳机音响"),
            ("苹果AirPods Pro 2代 主动降噪 无线充电盒 白色", "手机数码", "耳机音响"),
            ("BOSE QuietComfort 45 头戴式蓝牙降噪耳机 黑色", "手机数码", "耳机音响"),
            ("森海塞尔Momentum 4 Wireless 头戴式蓝牙耳机", "手机数码", "耳机音响"),
            ("JBL LIVE 660NC 头戴式无线蓝牙降噪耳机", "手机数码", "耳机音响"),
            
            # 服装鞋帽 - 男装
            ("优衣库 男装 圆领T恤 纯棉短袖 基础款 白色 L码", "服装鞋帽", "男装"),
            ("海澜之家 男士商务休闲西装套装 深蓝色 180/96A", "服装鞋帽", "男装"),
            ("李宁 男装 运动套装 春秋款 连帽卫衣+长裤 黑色", "服装鞋帽", "男装"),
            ("GXG 男装 修身牛仔裤 弹力小脚裤 深蓝色 32码", "服装鞋帽", "男装"),
            ("太平鸟 男装 潮流印花卫衣 宽松连帽上衣 灰色", "服装鞋帽", "男装"),
            
            # 服装鞋帽 - 女装
            ("ZARA 女装 碎花连衣裙 雪纺长裙 夏季新款 S码", "服装鞋帽", "女装"),
            ("H&M 女装 基础款T恤 纯棉短袖 白色 M码", "服装鞋帽", "女装"),
            ("ONLY 女装 高腰牛仔裤 修身显瘦 浅蓝色 27码", "服装鞋帽", "女装"),
            ("VERO MODA 女装 针织开衫 薄款外套 米色 S码", "服装鞋帽", "女装"),
            ("拉夏贝尔 女装 职业套装 西装外套+半身裙 黑色", "服装鞋帽", "女装"),
            
            # 服装鞋帽 - 运动鞋
            ("耐克 Air Max 270 男士跑步鞋 气垫运动鞋 黑白配色", "服装鞋帽", "运动鞋"),
            ("阿迪达斯 UltraBoost 22 女士跑步鞋 白色 36码", "服装鞋帽", "运动鞋"),
            ("新百伦 990v5 复古跑鞋 男女同款 灰色 经典款", "服装鞋帽", "运动鞋"),
            ("安踏 KT8 篮球鞋 汤普森签名款 实战球鞋 黑红", "服装鞋帽", "运动鞋"),
            ("李宁 音速9 CJ麦科勒姆签名篮球鞋 实战球鞋", "服装鞋帽", "运动鞋"),
            
            # 家居用品 - 家具
            ("宜家 MALM马尔姆 床架 白色 双人床 1.5米 北欧简约", "家居用品", "家具"),
            ("全友家居 现代简约沙发 布艺三人位 客厅家具 灰色", "家居用品", "家具"),
            ("红苹果 实木餐桌椅组合 橡木餐桌 1.4米 6人座", "家居用品", "家具"),
            ("顾家家居 真皮沙发 头层牛皮 现代简约 三人位", "家居用品", "家具"),
            ("林氏木业 北欧电视柜 现代简约 客厅储物柜 白色", "家居用品", "家具"),
            
            # 家居用品 - 厨房电器
            ("美的 电饭煲 IH电磁加热 4L容量 智能预约 MB-WFS4029", "家居用品", "厨房电器"),
            ("九阳 豆浆机 免滤豆浆机 1.3L 全自动 DJ13B-D88SG", "家居用品", "厨房电器"),
            ("苏泊尔 电压力锅 5L 智能预约 一键排气 SY-50YC9001Q", "家居用品", "厨房电器"),
            ("老板 抽油烟机 大吸力 侧吸式 自动清洗 CXW-200-26A7", "家居用品", "厨房电器"),
            ("方太 燃气灶 嵌入式 双灶 4.5kW大火力 JZT-FD21BE", "家居用品", "厨房电器"),
            
            # 美妆个护 - 护肤品
            ("兰蔻 小黑瓶精华液 30ml 修护精华 抗衰老 肌底液", "美妆个护", "护肤品"),
            ("雅诗兰黛 小棕瓶精华 50ml 修护精华 抗老紧致", "美妆个护", "护肤品"),
            ("SK-II 神仙水 230ml 护肤精华水 改善肌理", "美妆个护", "护肤品"),
            ("海蓝之谜 面霜 60ml 奢华修护 抗衰老面霜", "美妆个护", "护肤品"),
            ("资生堂 红腰子精华 30ml 修护精华 抗老紧致", "美妆个护", "护肤品"),
            
            # 美妆个护 - 彩妆
            ("雅诗兰黛 DW粉底液 30ml 持久遮瑕 自然色号", "美妆个护", "彩妆"),
            ("迪奥 999口红 3.5g 经典正红色 滋润质地", "美妆个护", "彩妆"),
            ("香奈儿 山茶花眼影盘 4色眼影 大地色系", "美妆个护", "彩妆"),
            ("阿玛尼 红管唇釉 6.5ml 丝绒哑光 正红色", "美妆个护", "彩妆"),
            ("兰蔻 菁纯口红 4.2g 滋润保湿 玫瑰色", "美妆个护", "彩妆"),
            
            # 食品饮料 - 零食坚果
            ("三只松鼠 坚果大礼包 混合装 1080g 每日坚果", "食品饮料", "零食坚果"),
            ("良品铺子 猪肉脯 靖江风味 108g 休闲零食", "食品饮料", "零食坚果"),
            ("百草味 夏威夷果 奶油味 100g 坚果炒货", "食品饮料", "零食坚果"),
            ("来伊份 蟹黄瓜子仁 108g 休闲零食 坚果炒货", "食品饮料", "零食坚果"),
            ("盐津铺子 鱼豆腐 麻辣味 106g 豆制品零食", "食品饮料", "零食坚果"),
            
            # 食品饮料 - 饮料
            ("可口可乐 经典款 330ml*24罐 碳酸饮料 整箱", "食品饮料", "饮料"),
            ("元气森林 无糖气泡水 白桃味 480ml*12瓶", "食品饮料", "饮料"),
            ("农夫山泉 天然水 550ml*24瓶 饮用水 整箱", "食品饮料", "饮料"),
            ("统一 绿茶 500ml*15瓶 茶饮料 整箱装", "食品饮料", "饮料"),
            ("王老吉 凉茶 310ml*24罐 植物饮料 整箱", "食品饮料", "饮料"),
        ]
        
        # 扩展数据集 - 添加更多变体
        extended_data = []
        for title, cat, subcat in sample_data:
            extended_data.append((title, cat, subcat))
            
            # 为每个商品创建一些变体
            variations = self._create_variations(title, cat, subcat)
            extended_data.extend(variations)
        
        # 创建DataFrame
        df = pd.DataFrame(extended_data, columns=['title', 'category', 'subcategory'])
        
        # 保存到文件
        filepath = os.path.join(self.data_dir, 'taobao_products.csv')
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"数据集已保存到: {filepath}, 共 {len(df)} 条记录")
        return filepath
    
    def _create_variations(self, title: str, category: str, subcategory: str) -> List[Tuple[str, str, str]]:
        """为商品标题创建变体"""
        variations = []
        
        # 简化版本（去掉一些修饰词）
        simplified = title.replace("新款", "").replace("经典款", "").replace("热销", "")
        if simplified != title:
            variations.append((simplified.strip(), category, subcategory))
        
        # 添加促销词汇
        promo_words = ["热销", "爆款", "新品", "特价", "包邮"]
        for word in promo_words[:2]:  # 只添加前两个，避免数据过多
            new_title = f"{word} {title}"
            variations.append((new_title, category, subcategory))
        
        return variations
    
    def load_dataset(self) -> pd.DataFrame:
        """加载数据集"""
        filepath = os.path.join(self.data_dir, 'taobao_products.csv')
        
        if not os.path.exists(filepath):
            logger.info("数据集不存在，正在创建...")
            self.download_sample_dataset()
        
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"数据集加载完成，共 {len(df)} 条记录")
        
        return df
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        df = self.load_dataset()
        
        stats = {
            'total_products': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'subcategories': df['subcategory'].value_counts().to_dict(),
            'category_count': df['category'].nunique(),
            'subcategory_count': df['subcategory'].nunique()
        }
        
        return stats

def prepare_training_data_from_dataset() -> Tuple[List[str], List[str], List[str]]:
    """从数据集准备训练数据"""
    loader = TaobaoDatasetLoader()
    df = loader.load_dataset()
    
    # 数据清洗
    df = df.dropna()
    df = df[df['title'].str.len() > 5]  # 过滤太短的标题
    
    texts = df['title'].tolist()
    categories = df['category'].tolist()
    subcategories = df['subcategory'].tolist()
    
    logger.info(f"准备训练数据完成: {len(texts)} 条记录")
    
    return texts, categories, subcategories
