"""
足球比赛预测策略分析工具

这个工具用于分析足球比赛预测模型的输出，并根据不同的概率分布模式应用不同的预测策略。

功能特点：
- 支持多种预测策略（单一预测、双重预测、特殊处理）
- 可配置的阈值参数
- 详细的预测结果分析
- 可视化预测分布
- 支持真实标签验证
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PredictionStrategy:
    """预测策略分析类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化预测策略
        
        Args:
            config: 配置字典，包含各种阈值参数
        """
        # 默认配置
        self.default_config = {
            'threshold_single': 0.5,      # 单一预测阈值
            'threshold_gap': 0.15,        # 概率差距阈值
            'threshold_sum': 0.79,        # 双重预测概率和阈值
            'draw_threshold': 0.34,       # 平局特殊处理阈值
            'reverse_threshold': 0.26,    # 反向预测阈值
            'low_prob_threshold': 0.18    # 低概率阈值
        }
        
        # 更新配置
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
    
    def predict(self, probs: List[float]) -> Tuple[List[int], str, Dict]:
        """根据概率分布进行预测
        
        Args:
            probs: 三个类别的概率 [home, draw, away]
            
        Returns:
            predictions: 预测的类别列表
            pred_type: 预测类型描述
            details: 预测详细信息
        """
        # 获取排序后的概率和对应的类别索引
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = np.array(probs)[sorted_indices]
        
        max_prob = sorted_probs[0]
        second_prob = sorted_probs[1]
        last_prob = sorted_probs[-1]
        prob_gap = max_prob - second_prob
        
        # 预测策略逻辑
        predictions, pred_type = self._apply_prediction_strategy(
            probs, sorted_indices, sorted_probs, max_prob, second_prob, last_prob, prob_gap
        )
        
        details = {
            'max_prob': max_prob,
            'second_prob': second_prob,
            'last_prob': last_prob,
            'prob_gap': prob_gap,
            'sorted_indices': sorted_indices.tolist(),
            'sorted_probs': sorted_probs.tolist()
        }
        
        return predictions, pred_type, details
    
    def _apply_prediction_strategy(self, probs, sorted_indices, sorted_probs, 
                                  max_prob, second_prob, last_prob, prob_gap):
        """应用预测策略"""
        
        # 策略1: 平局特殊处理1 - 当中间概率较高时
        if probs[1] >= self.config['draw_threshold']:
            return [1], "平局特殊处理1"
        
        # 策略2: 平局特殊处理2 - 当首尾概率相近且中间概率较低时
        if (probs[0] > 0.3 and probs[2] > 0.3 and 
            probs[0] < 0.4 and probs[2] < 0.4 and 
            abs(probs[0] - probs[2]) <= 0.08 and probs[1] <= 0.3):
            return [1], "平局特殊处理2"
        
        # 策略3: 平局特殊处理3 - 中间概率在特定范围内
        if 0.245 <= probs[1] <= 0.255:
            return [1], "平局特殊处理3"
        
        # 策略4: 平局特殊处理4 - 中间概率较低且差距小
        if (0.15 <= probs[1] <= 0.22 and (second_prob - last_prob) <= 0.02):
            return [1], "平局特殊处理4"
        
        # 策略5: 反向预测1 - 中间概率较高且首尾差距大
        if (probs[1] > self.config['reverse_threshold'] and 
            (abs(probs[0] - probs[2]) >= 0.28 or abs(probs[0] - probs[2]) <= 0.18)):
            if sorted_indices[0] == 0:
                return [2], "反向预测1"
            elif sorted_indices[0] == 2:
                return [0], "反向预测1"
        
        # 策略6: 反向预测2 - 中间和最后概率相近
        if -0.04 <= (probs[1] - probs[2]) <= 0:
            if sorted_indices[0] == 0:
                return [2], "反向预测2"
            elif sorted_indices[0] == 2:
                return [0], "反向预测2"
        
        # 策略7: 反向预测3 - 中间概率在特定范围
        if 0.22 <= probs[1] <= 0.23:
            if sorted_indices[0] == 0:
                return [2], "反向预测3"
            elif sorted_indices[0] == 2:
                return [0], "反向预测3"
        
        # 策略8: 双重预测 - 前两个概率和较高
        if (max_prob + second_prob) >= self.config['threshold_sum']:
            return [sorted_indices[0], sorted_indices[1]], "双重预测"
        
        # 策略9: 单一预测 - 最大概率较高且差距明显
        if (max_prob >= self.config['threshold_single'] and 
            prob_gap >= self.config['threshold_gap']):
            return [sorted_indices[0]], "单一预测"
        
        # 默认策略: 选择最大概率
        return [sorted_indices[0]], "默认单一预测"
    
    def evaluate_predictions(self, df: pd.DataFrame, true_labels: Optional[List] = None) -> Dict:
        """评估预测结果
        
        Args:
            df: 包含概率数据的DataFrame
            true_labels: 真实标签列表（可选）
            
        Returns:
            评估结果字典
        """
        results = []
        correct = 0
        total = len(df)
        
        prediction_types = {}
        
        for idx, row in df.iterrows():
            probs = [row['class0'], row['class1'], row['class2']]
            
            # 进行预测
            predictions, pred_type, details = self.predict(probs)
            
            # 检查是否有真实标签
            true_label = None
            is_correct = None
            
            if true_labels and idx < len(true_labels):
                true_label = true_labels[idx]
                is_correct = true_label in predictions
                if is_correct:
                    correct += 1
            
            # 统计预测类型
            if pred_type not in prediction_types:
                prediction_types[pred_type] = 0
            prediction_types[pred_type] += 1
            
            # 保存结果
            result = {
                'index': idx + 1,
                'predictions': predictions,
                'prediction_type': pred_type,
                'max_prob': details['max_prob'],
                'second_prob': details['second_prob'],
                'prob_gap': details['prob_gap'],
                'true_label': true_label,
                'is_correct': is_correct
            }
            results.append(result)
        
        # 计算准确率
        accuracy = correct / total if total > 0 else 0
        
        return {
            'detailed_results': results,
            'summary': {
                'total_samples': total,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'prediction_types': prediction_types
            }
        }


def get_sample_data() -> pd.DataFrame:
    """获取示例数据"""
    data = [
        [0.2234, 0.3562, 0.4204],
        [0.5249, 0.2244, 0.2506],
        [0.4450, 0.2268, 0.3283],
        [0.1884, 0.2577, 0.5539],
        [0.2612, 0.3017, 0.4371],
        [0.6430, 0.1815, 0.1754],
        [0.6069, 0.2125, 0.1806],
        [0.5983, 0.2026, 0.1991],
        [0.2196, 0.1928, 0.5876],
        [0.3428, 0.2075, 0.4497],
        [0.5089, 0.1817, 0.3094],
        [0.3636, 0.2620, 0.3743],
        [0.2373, 0.1926, 0.5701],
        [0.1937, 0.2544, 0.5520],
        [0.4933, 0.2570, 0.2497],
        [0.5702, 0.2070, 0.2228],
        [0.3995, 0.2823, 0.3182]
    ]
    return pd.DataFrame(data, columns=['class0', 'class1', 'class2'])


def visualize_predictions(evaluation):
    """可视化预测结果"""
    detailed_results = pd.DataFrame(evaluation['detailed_results'])
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 预测类型分布
    pred_type_counts = detailed_results['prediction_type'].value_counts()
    axes[0, 0].pie(pred_type_counts.values, labels=pred_type_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('预测类型分布')
    
    # 2. 最大概率分布
    axes[0, 1].hist(detailed_results['max_prob'], bins=20, alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('最大概率')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('最大概率分布')
    
    # 3. 概率差距分布
    axes[1, 0].hist(detailed_results['prob_gap'], bins=20, alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('概率差距')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('概率差距分布')
    
    # 4. 预测类型与最大概率的关系
    for pred_type in detailed_results['prediction_type'].unique():
        subset = detailed_results[detailed_results['prediction_type'] == pred_type]
        axes[1, 1].scatter(subset['max_prob'], subset['second_prob'], 
                           label=pred_type, alpha=0.7)
    
    axes[1, 1].set_xlabel('最大概率')
    axes[1, 1].set_ylabel('第二概率')
    axes[1, 1].set_title('最大概率 vs 第二概率')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def test_different_configs(df):
    """测试不同配置的效果"""
    
    # 不同配置
    configs = {
        '保守配置': {
            'threshold_single': 0.6,
            'threshold_gap': 0.2,
            'threshold_sum': 0.85,
            'draw_threshold': 0.4
        },
        '激进配置': {
            'threshold_single': 0.4,
            'threshold_gap': 0.1,
            'threshold_sum': 0.75,
            'draw_threshold': 0.3
        },
        '默认配置': {}
    }
    
    results = {}
    
    for config_name, config in configs.items():
        strategy = PredictionStrategy(config)
        evaluation = strategy.evaluate_predictions(df)
        
        results[config_name] = {
            'total_samples': evaluation['summary']['total_samples'],
            'prediction_types': evaluation['summary']['prediction_types']
        }
    
    # 显示结果比较
    print("=== 不同配置的预测结果比较 ===")
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  总样本数: {result['total_samples']}")
        print(f"  预测类型分布:")
        for pred_type, count in result['prediction_types'].items():
            percentage = (count / result['total_samples']) * 100
            print(f"    {pred_type}: {count} ({percentage:.1f}%)")
    
    return results


def test_with_true_labels(df):
    """使用真实标签进行测试"""
    
    # 示例真实标签（这里需要根据实际情况调整）
    true_labels = [2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 1, 0, 2, 2]
    
    # 创建策略实例
    strategy = PredictionStrategy()
    
    # 执行评估
    evaluation = strategy.evaluate_predictions(df, true_labels)
    
    # 显示结果
    summary = evaluation['summary']
    print("=== 真实标签测试结果 ===")
    print(f"总样本数: {summary['total_samples']}")
    print(f"正确预测数: {summary['correct_predictions']}")
    print(f"准确率: {summary['accuracy']:.2%}")
    
    # 按预测类型分析准确率
    detailed_results = pd.DataFrame(evaluation['detailed_results'])
    print("\n各预测类型的准确率:")
    for pred_type in detailed_results['prediction_type'].unique():
        subset = detailed_results[detailed_results['prediction_type'] == pred_type]
        if len(subset) > 0:
            accuracy = subset['is_correct'].mean()
            count = len(subset)
            print(f"  {pred_type}: {accuracy:.2%} ({count}个样本)")
    
    return evaluation


def main():
    """主函数 - 演示使用方法"""
    
    # 获取示例数据
    df = get_sample_data()
    print(f"数据形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())
    
    # 创建预测策略实例
    strategy = PredictionStrategy()
    
    # 执行预测分析
    evaluation = strategy.evaluate_predictions(df)
    
    # 打印总体结果
    summary = evaluation['summary']
    print("\n=== 预测分析结果 ===")
    print(f"总样本数: {summary['total_samples']}")
    print(f"预测类型分布:")
    for pred_type, count in summary['prediction_types'].items():
        percentage = (count / summary['total_samples']) * 100
        print(f"  {pred_type}: {count} ({percentage:.1f}%)")
    
    # 显示详细结果
    detailed_results = pd.DataFrame(evaluation['detailed_results'])
    print("\n=== 详细预测结果 ===")
    print(detailed_results[['index', 'predictions', 'prediction_type', 'max_prob', 'second_prob']].head(10))
    
    # 可视化结果
    visualize_predictions(evaluation)
    
    # 测试不同配置
    config_results = test_different_configs(df)
    
    # 测试真实标签
    true_label_evaluation = test_with_true_labels(df)
    
    return evaluation, config_results, true_label_evaluation


if __name__ == "__main__":
    # 运行主函数
    evaluation, config_results, true_label_evaluation = main() 