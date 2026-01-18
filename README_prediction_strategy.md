# 足球比赛预测策略分析工具

## 概述

这是一个用于分析足球比赛预测模型输出的工具，能够根据不同的概率分布模式应用不同的预测策略。该工具支持多种预测策略，包括单一预测、双重预测和各种特殊处理策略。

## 功能特点

- ✅ **多种预测策略**：支持9种不同的预测策略
- ✅ **可配置参数**：所有阈值参数都可以自定义
- ✅ **详细分析**：提供详细的预测结果分析
- ✅ **可视化**：内置多种可视化图表
- ✅ **真实标签验证**：支持使用真实标签进行准确率评估
- ✅ **批量处理**：支持批量预测和分析
- ✅ **配置比较**：可以比较不同配置的效果

## 文件结构

```
├── prediction_strategy_improved.py    # 主要工具类
├── prediction_strategy_demo.ipynb     # 使用演示
├── prediction_strategy.ipynb          # 原始文件
└── README_prediction_strategy.md      # 说明文档
```

## 安装依赖

```bash
pip install numpy pandas matplotlib seaborn
```

## 快速开始

### 1. 基本使用

```python
from prediction_strategy_improved import PredictionStrategy

# 创建预测策略实例
strategy = PredictionStrategy()

# 对单个样本进行预测
probs = [0.4, 0.3, 0.3]  # [home, draw, away]
predictions, pred_type, details = strategy.predict(probs)
print(f"预测结果: {predictions}")
print(f"预测类型: {pred_type}")
```

### 2. 批量预测

```python
import pandas as pd
from prediction_strategy_improved import PredictionStrategy, get_sample_data

# 获取示例数据
df = get_sample_data()

# 创建策略实例
strategy = PredictionStrategy()

# 批量预测
evaluation = strategy.evaluate_predictions(df)

# 查看结果
summary = evaluation['summary']
print(f"总样本数: {summary['total_samples']}")
print(f"预测类型分布: {summary['prediction_types']}")
```

### 3. 自定义配置

```python
# 自定义配置
config = {
    'threshold_single': 0.6,    # 单一预测阈值
    'threshold_gap': 0.2,       # 概率差距阈值
    'threshold_sum': 0.85,      # 双重预测概率和阈值
    'draw_threshold': 0.4,      # 平局特殊处理阈值
    'reverse_threshold': 0.26,  # 反向预测阈值
    'low_prob_threshold': 0.18  # 低概率阈值
}

strategy = PredictionStrategy(config)
```

## 预测策略详解

### 1. 平局特殊处理策略

- **平局特殊处理1**：当中间概率 ≥ 0.34 时，预测平局
- **平局特殊处理2**：当首尾概率相近且中间概率较低时，预测平局
- **平局特殊处理3**：当中间概率在 0.245-0.255 范围内时，预测平局
- **平局特殊处理4**：当中间概率较低且差距小时，预测平局

### 2. 反向预测策略

- **反向预测1**：当中间概率较高且首尾差距大时，进行反向预测
- **反向预测2**：当中间和最后概率相近时，进行反向预测
- **反向预测3**：当中间概率在特定范围时，进行反向预测

### 3. 标准预测策略

- **双重预测**：当前两个概率和 ≥ 0.79 时，预测两个结果
- **单一预测**：当最大概率 ≥ 0.5 且差距 ≥ 0.15 时，预测单一结果
- **默认单一预测**：选择最大概率作为预测结果

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold_single` | 0.5 | 单一预测阈值 |
| `threshold_gap` | 0.15 | 概率差距阈值 |
| `threshold_sum` | 0.79 | 双重预测概率和阈值 |
| `draw_threshold` | 0.34 | 平局特殊处理阈值 |
| `reverse_threshold` | 0.26 | 反向预测阈值 |
| `low_prob_threshold` | 0.18 | 低概率阈值 |

## 使用示例

### 示例1：基本预测

```python
from prediction_strategy_improved import PredictionStrategy

strategy = PredictionStrategy()

# 测试不同的概率分布
test_cases = [
    [0.4, 0.3, 0.3],  # 接近均匀分布
    [0.6, 0.2, 0.2],  # 明显偏向第一个
    [0.2, 0.5, 0.3],  # 中间概率较高
    [0.3, 0.2, 0.5],  # 偏向最后一个
]

for i, probs in enumerate(test_cases):
    predictions, pred_type, details = strategy.predict(probs)
    print(f"案例 {i+1}: {probs} -> {predictions} ({pred_type})")
```

### 示例2：批量分析

```python
from prediction_strategy_improved import PredictionStrategy, get_sample_data, visualize_predictions

# 获取数据
df = get_sample_data()

# 创建策略
strategy = PredictionStrategy()

# 批量预测
evaluation = strategy.evaluate_predictions(df)

# 可视化结果
visualize_predictions(evaluation)
```

### 示例3：真实标签验证

```python
# 假设我们有真实标签
true_labels = [2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 1, 0, 2, 2]

# 使用真实标签进行评估
evaluation = strategy.evaluate_predictions(df, true_labels)

# 查看准确率
summary = evaluation['summary']
print(f"准确率: {summary['accuracy']:.2%}")
```

## 可视化功能

工具提供了多种可视化功能：

1. **预测类型分布饼图**：显示各种预测策略的使用比例
2. **最大概率分布直方图**：分析最大概率的分布情况
3. **概率差距分布直方图**：分析概率差距的分布情况
4. **最大概率 vs 第二概率散点图**：分析不同预测类型的概率特征

## 配置建议

### 保守配置
适用于对准确性要求较高的场景：
```python
conservative_config = {
    'threshold_single': 0.6,
    'threshold_gap': 0.2,
    'threshold_sum': 0.85,
    'draw_threshold': 0.4
}
```

### 激进配置
适用于需要更多样化预测的场景：
```python
aggressive_config = {
    'threshold_single': 0.4,
    'threshold_gap': 0.1,
    'threshold_sum': 0.75,
    'draw_threshold': 0.3
}
```

### 平衡配置
默认配置，适合大多数场景：
```python
default_config = {
    'threshold_single': 0.5,
    'threshold_gap': 0.15,
    'threshold_sum': 0.79,
    'draw_threshold': 0.34
}
```

## 注意事项

1. **数据格式**：输入数据应为包含 `class0`, `class1`, `class2` 列的DataFrame
2. **概率归一化**：确保三个概率的和为1
3. **标签映射**：0=主队胜，1=平局，2=客队胜
4. **配置调优**：根据实际数据特点调整配置参数

## 故障排除

### 常见问题

1. **导入错误**：确保已安装所有依赖包
2. **数据格式错误**：检查DataFrame列名是否正确
3. **可视化显示问题**：确保matplotlib配置正确

### 调试建议

1. 使用小数据集测试
2. 检查概率值是否在[0,1]范围内
3. 验证概率和为1
4. 逐步调整配置参数

## 更新日志

- **v2.0**: 重构代码结构，增加可视化功能
- **v1.0**: 初始版本，基本预测功能

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。

## 许可证

MIT License 