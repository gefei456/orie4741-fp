#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : orie4741-fp
@File    : multi_feature_model.py
@IDE     : PyCharm
@Author  : wgf
@Data    : 2024/10/29 21:36
@Description:
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionLayer(nn.Module):
    def __init__(self, input_size):
        super(InteractionLayer, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x):
        # 特征交互
        interactions = torch.bmm(x.unsqueeze(1), x.unsqueeze(2)).view(x.size(0), -1)
        return F.relu(self.linear(interactions))


class MultiFeatureModel(nn.Module):
    def __init__(self, num_classes, numerical_input_size, sequence_input_size, num_categories, embedding_dim):
        super(MultiFeatureModel, self).__init__()

        # 数值特征处理
        self.fc_numerical = nn.Sequential(
            nn.Linear(numerical_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 序列特征处理 (使用 GRU)
        self.gru = nn.GRU(input_size=sequence_input_size, hidden_size=64, batch_first=True)

        # 类别特征处理 (使用嵌入层)
        self.embedding = nn.Embedding(num_categories, embedding_dim)

        # 特征交互层
        self.interaction_layer = InteractionLayer(64 * 3)  # 假设后续特征维度为 64 * 3

        # 输出层
        self.fc_output = nn.Linear(64 + 64 + 64 + (64 * 3), num_classes)  # 包括交互特征维度

    def forward(self, numerical_features, sequence_features, categorical_features):
        # 处理数值特征
        numerical_out = self.fc_numerical(numerical_features)

        # 处理序列特征
        seq_out, _ = self.gru(sequence_features)
        seq_out = seq_out[:, -1, :]  # 取最后的时刻输出

        # 处理类别特征
        categorical_out = self.embedding(categorical_features).view(categorical_features.size(0), -1)

        # 融合输出
        combined = torch.cat((numerical_out, seq_out, categorical_out), dim=1)

        # 特征交互
        interaction_out = self.interaction_layer(combined)

        # 最终输出
        final_out = torch.cat((numerical_out, seq_out, categorical_out, interaction_out), dim=1)
        output = self.fc_output(final_out)

        return output
