# prompts/classifier_prompts.py
from . import YA_MCPServer_Prompt
from mcp.types import PromptMessage, TextContent

@YA_MCPServer_Prompt(
    name="classifier_recommendation",
    title="分类器推荐方案",
    description="生成面向用户的模型推荐话术与执行方案"
)
def classifier_recommendation_prompt(user_requirement: str) -> list[PromptMessage]:
    """生成分类器推荐方案"""
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""作为AI分类算法专家，请根据以下用户需求生成详细的模型推荐方案：

用户需求：{user_requirement}

请按以下结构输出：

1. 需求分析
   - 任务类型识别
   - 数据规模估算
   - 性能要求评估

2. 模型推荐
   - 推荐算法（排序）
   - 选择理由
   - 预期效果

3. 数据预处理步骤
   - 数据清洗方案
   - 特征工程建议
   - 数据增强策略

4. 模型评估建议
   - 评估指标选择
   - 验证方法
   - 优化方向

5. 实施计划
   - 开发步骤
   - 时间估算
   - 风险提示"""
            )
        )
    ]

@YA_MCPServer_Prompt(
    name="data_analysis_report",
    title="数据分析报告",
    description="生成数据预处理与特征工程的分析报告"
)
def data_analysis_report_prompt(dataset_info: str) -> list[PromptMessage]:
    """生成数据分析报告"""
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""请基于以下数据集信息生成详细的数据分析报告：

数据集信息：
{dataset_info}

报告应包含：

## 数据概览
- 样本数量与特征维度
- 数据类型分布
- 缺失值统计

## 数据质量分析
- 数据完整性
- 异常值检测
- 类别平衡性

## 特征分析
- 关键特征识别
- 特征相关性
- 特征重要性排序

## 预处理建议
- 缺失值处理方案
- 特征编码策略
- 数据标准化建议

## 可视化建议
- 推荐的图表类型
- 关键指标展示

请以Markdown格式输出，包含表格和列表。"""
            )
        )
    ]

@YA_MCPServer_Prompt(
    name="model_explain",
    title="模型可解释性说明",
    description="生成模型决策过程的可解释性说明"
)
def model_explain_prompt(model_name: str, prediction: str, confidence: float) -> list[PromptMessage]:
    """生成模型解释"""
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"""请解释以下分类模型的决策过程：

模型：{model_name}
预测结果：{prediction}
置信度：{confidence:.2%}

请从以下角度进行解释：

1. 决策路径
   - 模型如何处理输入特征
   - 关键决策节点
   - 最终分类依据

2. 特征贡献度
   - 哪些特征对结果影响最大
   - 特征权重分析
   - 特征交互效应

3. 置信度分析
   - 置信度水平解读
   - 不确定性来源
   - 可信度评估

4. 业务解读
   - 结果的实际意义
   - 应用建议
   - 注意事项

请用通俗易懂的语言解释，避免过多技术术语。"""
            )
        )
    ]
