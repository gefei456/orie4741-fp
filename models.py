import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义 MLP 模型
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[512, 256, 128], output_dim=3, dropout_prob=0.3):
        super(MLPModel, self).__init__()

        layers = []
        in_dim = input_dim
        
        # 添加批归一化层
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 每个线性层后添加批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = hidden_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        # 输入层批归一化
        x = self.batch_norm_input(x)
        # 通过隐藏层
        x = self.hidden_layers(x)
        # 输出层（不使用激活函数，因为会在损失函数中使用CrossEntropyLoss）
        return self.output_layer(x)

# 自定义的损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 定义训练过程
def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=128, 
                learning_rate=1e-3, early_stopping=True, patience=20):
    # 创建优化器，使用 AdamW 和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = FocalLoss()  # 使用 Focal Loss

    # 批次处理数据
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化早期停止相关变量
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_weights = None

    # 用于记录训练损失和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))  # 使用 argmax 因为标签是 one-hot 编码
            running_train_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 计算训练集上的平均损失
        train_loss = running_train_loss / len(dataloader)

        # 计算验证集上的损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, torch.argmax(y_val, dim=1))

        # 更新学习率
        scheduler.step(val_loss)

        # 保存训练损失和验证损失
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())

        # 打印当前的训练损失和验证损失
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 早期停止：检查验证损失
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

    # 恢复最好的模型权重
    if early_stopping and best_model_weights:
        model.load_state_dict(best_model_weights)

    # 绘制训练损失和验证损失的曲线
    plot_loss_curve(train_losses, val_losses)

    return model

# 计算 top-k 准确率
def compute_top_k_accuracy(model, x_test, y_test, k=2):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        # 对于 one-hot 编码的标签，需要先转换回类别索引
        y_test_indices = torch.argmax(y_test, dim=1)
        _, top_k_indices = torch.topk(outputs, k, dim=1, largest=True, sorted=True)
        correct_top_1 = torch.sum(top_k_indices[:, 0] == y_test_indices).item()
        correct_top_2 = torch.sum((top_k_indices[:, 0] == y_test_indices) | (top_k_indices[:, 1] == y_test_indices)).item()
        top_1_accuracy = correct_top_1 / len(y_test)
        top_2_accuracy = correct_top_2 / len(y_test)
    return top_1_accuracy, top_2_accuracy

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', linestyle='--', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show() 