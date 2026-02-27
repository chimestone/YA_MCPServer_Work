# tools/product_classifier_models.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
import pickle
import jieba
from modules.YA_Common.utils.logger import get_logger

logger = get_logger("ProductClassifierModels")

class ProductClassifierEnsemble:
    """商品分类集成模型"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, tokenizer=self._tokenize)
        self.models = {
            'svm': SVC(kernel='linear', probability=True),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        self.is_trained = False
    
    @staticmethod
    def _tokenize(text):
        """中文分词"""
        return jieba.lcut(text)
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """训练所有模型"""
        X = self.vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[name] = round(score, 4)
            logger.info(f"{name} 训练完成，准确率: {score:.4f}")
        
        self.is_trained = True
        return results
    
    def predict(self, text: str, algorithms: List[str]) -> Dict[str, Dict[str, Any]]:
        """预测商品类别"""
        X = self.vectorizer.transform([text])
        results = {}
        
        for algo in algorithms:
            if algo == 'ensemble':
                predictions = []
                confidences = []
                for model in self.models.values():
                    pred = model.predict(X)[0]
                    conf = model.predict_proba(X).max()
                    predictions.append(pred)
                    confidences.append(conf)
                
                final_pred = max(set(predictions), key=predictions.count)
                final_conf = sum(confidences) / len(confidences)
                results['ensemble'] = {'prediction': final_pred, 'confidence': final_conf}
            elif algo in self.models:
                model = self.models[algo]
                pred = model.predict(X)[0]
                conf = model.predict_proba(X).max()
                results[algo] = {'prediction': pred, 'confidence': conf}
        
        return results
    
    def save_model(self, path: str) -> bool:
        """保存模型"""
        try:
            with open(path, 'wb') as f:
                pickle.dump({'vectorizer': self.vectorizer, 'models': self.models}, f)
            logger.info(f"模型已保存到: {path}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """加载模型"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.models = data['models']
            self.is_trained = True
            logger.info(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
