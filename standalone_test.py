# standalone_test.py
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 简化版的数据加载器
class SimpleDatasetLoader:
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_sample_data(self) -> pd.DataFrame:
        sample_data = [
            # 手机数码类
            ("苹果iPhone 15 Pro 256GB 深空黑色 5G手机", "手机数码", "智能手机"),
            ("华为Mate60 Pro 512GB 雅川青 国产芯片", "手机数码", "智能手机"),
            ("小米13 Ultra 徕卡影像 16GB+1TB 橄榄绿", "手机数码", "智能手机"),
            ("联想ThinkPad X1 Carbon 14英寸轻薄笔记本", "手机数码", "笔记本电脑"),
            ("戴尔XPS 13 超轻薄本 11代酷睿i7", "手机数码", "笔记本电脑"),
            ("索尼WH-1000XM5 无线降噪耳机 黑色", "手机数码", "耳机音响"),
            
            # 服装鞋帽类
            ("优衣库男装圆领T恤 纯棉短袖 白色L码", "服装鞋帽", "男装"),
            ("ZARA女士连衣裙 夏季新款 碎花雪纺", "服装鞋帽", "女装"),
            ("耐克Air Max 270 男士跑步鞋 黑白配色", "服装鞋帽", "运动鞋"),
            ("阿迪达斯三叶草经典款 女士休闲鞋", "服装鞋帽", "休闲鞋"),
            ("Coach蔻驰女士手提包 真皮单肩包", "服装鞋帽", "箱包"),
            
            # 家居用品类
            ("宜家MALM床架 白色双人床 1.5米", "家居用品", "家具"),
            ("美的电饭煲 4L智能预约 IH加热", "家居用品", "厨房电器"),
            ("戴森V15 Detect无绳吸尘器 激光探测", "家居用品", "生活电器"),
            ("全棉时代纯棉四件套 1.8米床 简约风", "家居用品", "床上用品"),
            
            # 美妆个护类
            ("兰蔻小黑瓶精华液 30ml 抗衰老", "美妆个护", "护肤品"),
            ("雅诗兰黛DW粉底液 持久遮瑕 30ml", "美妆个护", "彩妆"),
            ("海飞丝去屑洗发水 750ml 清爽型", "美妆个护", "洗护用品"),
            
            # 食品饮料类
            ("三只松鼠坚果大礼包 混合装1080g", "食品饮料", "零食坚果"),
            ("茅台飞天53度 500ml 白酒礼盒装", "食品饮料", "酒类"),
            ("元气森林无糖气泡水 白桃味 12瓶", "食品饮料", "饮料"),
        ]
        
        # 扩展数据
        extended_data = []
        for title, cat, subcat in sample_data:
            extended_data.append((title, cat, subcat))
            # 添加一些变体
            extended_data.append((f"热销 {title}", cat, subcat))
            extended_data.append((f"新品 {title}", cat, subcat))
        
        df = pd.DataFrame(extended_data, columns=['title', 'category', 'subcategory'])
        return df

# 简化版分类器
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import jieba

class SimpleProductClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.svm = SVC(probability=True, random_state=42)
        self.nb = MultinomialNB()
        self.rf = RandomForestClassifier(n_estimators=50, random_state=42)
        self.ensemble = VotingClassifier([
            ('svm', self.svm),
            ('nb', self.nb),
            ('rf', self.rf)
        ], voting='soft')
        self.is_trained = False
        self.label_encoder = {}
        self.reverse_label_encoder = {}
    
    def preprocess_text(self, text: str) -> str:
        try:
            words = jieba.cut(text)
            return ' '.join(words)
        except:
            return text
    
    def train(self, texts: List[str], labels: List[str]):
        print("开始训练模型...")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
        
        # 编码标签
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        y = np.array([self.label_encoder[label] for label in labels])
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        self.svm.fit(X_train, y_train)
        self.nb.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)
        self.ensemble.fit(X_train, y_train)
        
        # 评估
        svm_acc = accuracy_score(y_test, self.svm.predict(X_test))
        nb_acc = accuracy_score(y_test, self.nb.predict(X_test))
        rf_acc = accuracy_score(y_test, self.rf.predict(X_test))
        ensemble_acc = accuracy_score(y_test, self.ensemble.predict(X_test))
        
        print(f"SVM准确率: {svm_acc:.3f}")
        print(f"朴素贝叶斯准确率: {nb_acc:.3f}")
        print(f"随机森林准确率: {rf_acc:.3f}")
        print(f"集成模型准确率: {ensemble_acc:.3f}")
        
        self.is_trained = True
        return {
            'svm': svm_acc,
            'naive_bayes': nb_acc,
            'random_forest': rf_acc,
            'ensemble': ensemble_acc
        }
    
    def predict(self, text: str):
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # 各算法预测
        svm_pred = self.svm.predict(X)[0]
        svm_proba = self.svm.predict_proba(X)[0].max()
        
        nb_pred = self.nb.predict(X)[0]
        nb_proba = self.nb.predict_proba(X)[0].max()
        
        rf_pred = self.rf.predict(X)[0]
        rf_proba = self.rf.predict_proba(X)[0].max()
        
        ensemble_pred = self.ensemble.predict(X)[0]
        ensemble_proba = self.ensemble.predict_proba(X)[0].max()
        
        return {
            'svm': {
                'prediction': self.reverse_label_encoder[svm_pred],
                'confidence': svm_proba
            },
            'naive_bayes': {
                'prediction': self.reverse_label_encoder[nb_pred],
                'confidence': nb_proba
            },
            'random_forest': {
                'prediction': self.reverse_label_encoder[rf_pred],
                'confidence': rf_proba
            },
            'ensemble': {
                'prediction': self.reverse_label_encoder[ensemble_pred],
                'confidence': ensemble_proba
            }
        }

def main():
    print("=== 商品分类器独立测试 ===\n")
    
    # 1. 准备数据
    loader = SimpleDatasetLoader()
    df = loader.get_sample_data()
    
    print(f"数据集信息:")
    print(f"   总商品数: {len(df)}")
    print(f"   类别数: {df['category'].nunique()}")
    print(f"   类别分布: {dict(df['category'].value_counts())}")
    
    # 2. 训练模型
    classifier = SimpleProductClassifier()
    texts = df['title'].tolist()
    categories = df['category'].tolist()
    
    print(f"\n开始训练...")
    results = classifier.train(texts, categories)
    
    # 3. 测试分类
    print(f"\n商品分类测试:")
    test_products = [
        "华为P60 Pro 256GB 羽砂紫 5G手机 徕卡影像",
        "Nike Air Jordan 1 High OG 男士篮球鞋 芝加哥配色",
        "戴森V12 Detect Slim无绳吸尘器 激光探测科技",
        "雅诗兰黛小棕瓶眼部精华 15ml 抗衰老眼霜",
        "三只松鼠每日坚果 混合坚果 30包装 健康零食"
    ]
    
    for product in test_products:
        print(f"\n   测试商品: {product}")
        try:
            result = classifier.predict(product)
            ensemble_result = result['ensemble']
            print(f"   分类结果: {ensemble_result['prediction']}")
            print(f"   置信度: {ensemble_result['confidence']:.3f}")
            
            # 显示各算法结果
            print(f"   各算法结果:")
            for algo, res in result.items():
                if algo != 'ensemble':
                    print(f"     {algo}: {res['prediction']} ({res['confidence']:.3f})")
                    
        except Exception as e:
            print(f"   分类失败: {e}")

if __name__ == "__main__":
    main()
