# test_real_data_classifier.py
from tools.product_classifier_enhanced import (
    classify_product_with_real_data, 
    get_dataset_statistics, 
    evaluate_model_performance
)

def main():
    print("=== 基于真实数据的商品分类器测试 ===\n")
    
    # 1. 获取数据集统计信息
    print("1. 数据集统计信息:")
    stats = get_dataset_statistics()
    print(f"   总商品数: {stats['dataset_overview']['total_products']}")
    print(f"   类别数: {stats['dataset_overview']['unique_categories']}")
    print(f"   子类别数: {stats['dataset_overview']['unique_subcategories']}")
    print(f"   平均标题长度: {stats['data_quality']['average_title_length']} 字符")
    print(f"   数据平衡性: {stats['data_quality']['category_balance']}")
    
    print("\n类别分布:")
    for category, count in stats['category_distribution'].items():
        print(f"   {category}: {count} 个商品")
    
    # 2. 测试商品分类
    print("\n2. 商品分类测试:")
    test_products = [
        "华为P60 Pro 256GB 羽砂紫 5G手机 徕卡影像",
        "Nike Air Jordan 1 High OG 男士篮球鞋 芝加哥配色",
        "戴森V12 Detect Slim无绳吸尘器 激光探测科技",
        "雅诗兰黛小棕瓶眼部精华 15ml 抗衰老眼霜",
        "三只松鼠每日坚果 混合坚果 30包装 健康零食"
    ]
    
    for product in test_products:
        print(f"\n   测试商品: {product}")
        result = classify_product_with_real_data(product)
        
        if 'error' not in result:
            pred = result['prediction']
            print(f"   分类结果: {pred['category']}")
            print(f"   置信度: {pred['confidence']} ({pred['confidence_level']})")
            
            # 显示算法一致性
            if 'confidence_analysis' in result['analysis']:
                consensus = result['analysis']['confidence_analysis']['prediction_consensus']
                consensus_rate = result['analysis']['confidence_analysis']['consensus_rate']
                print(f"   算法一致性: {'完全一致' if consensus else f'一致率 {consensus_rate:.2%}'}")
        else:
            print(f"   错误: {result['error']}")
    
    # 3. 模型性能评估
    print("\n3. 模型性能评估:")
    performance = evaluate_model_performance()
    
    if 'error' not in performance:
        print(f"   评估样本数: {performance['evaluation_summary']['sample_size']}")
        print(f"   最佳算法: {performance['best_algorithm']}")
        
        print("\n   各算法准确率:")
        for algo, perf in performance['algorithm_performance'].items():
            print(f"   {algo}: {perf['accuracy']:.3f} ({perf['correct_predictions']}/{perf['total_predictions']})")
    else:
        print(f"   评估失败: {performance['error']}")

if __name__ == "__main__":
    main()
