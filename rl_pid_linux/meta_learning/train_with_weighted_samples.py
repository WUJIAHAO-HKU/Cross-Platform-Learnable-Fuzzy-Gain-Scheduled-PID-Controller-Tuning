#!/usr/bin/env python3
"""
使用加权样本训练元学习PID网络
根据优化误差自动分配样本权重
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================================================================
# SimplePIDPredictor（与之前保持一致）
# ============================================================================
class SimplePIDPredictor(nn.Module):
    """简单的MLP预测单组PID参数"""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# 加权损失函数
# ============================================================================
class WeightedMSELoss(nn.Module):
    """加权均方误差损失"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weights):
        """
        Args:
            pred: 预测值 (N, 3)
            target: 真实值 (N, 3)
            weights: 样本权重 (N,)
        """
        # 计算每个样本的MSE
        mse = ((pred - target) ** 2).mean(dim=1)  # (N,)
        
        # 加权平均
        weighted_mse = (mse * weights).sum() / weights.sum()
        
        return weighted_mse


# ============================================================================
# 样本权重计算
# ============================================================================
def compute_sample_weights(optimization_errors, weight_strategy='threshold'):
    """
    计算样本权重
    
    Args:
        optimization_errors: 优化误差列表（度数）
        weight_strategy: 权重策略
            - 'inverse': w = 1 / (1 + error/5) - 严格反比
            - 'exponential': w = exp(-error / 15) - 严格指数衰减
            - 'threshold': 三档权重 <20°→1.0, 20-35°→0.5, ≥35°→0.05
            - 'strict': 只用误差<25°的样本，其余权重0
    
    Returns:
        weights: 归一化的权重
    """
    errors = np.array(optimization_errors)
    
    if weight_strategy == 'inverse':
        # 反比权重：误差越小，权重越大（更严格）
        weights = 1.0 / (1.0 + errors / 5.0)  # 除以5缩放（之前10，现在更严格）
    
    elif weight_strategy == 'exponential':
        # 指数权重：更激进地降低大误差样本权重（更严格）
        weights = np.exp(-errors / 15.0)  # 15比之前的20更严格
    
    elif weight_strategy == 'threshold':
        # 阈值权重：误差过大的样本降权（更严格）
        threshold_high_quality = 20.0  # 高质量阈值
        threshold_acceptable = 35.0     # 可接受阈值
        
        # 三档权重：优秀(1.0), 良好(0.5), 差(0.05)
        weights = np.where(errors < threshold_high_quality, 1.0,
                  np.where(errors < threshold_acceptable, 0.5, 0.05))
    
    elif weight_strategy == 'strict':
        # 最严格：只用高质量样本，其余完全排除
        strict_threshold = 25.0
        weights = np.where(errors < strict_threshold, 1.0, 0.0)
        
        n_excluded = (weights == 0).sum()
        print(f"   ⚠️  strict模式：排除{n_excluded}个样本（误差≥{strict_threshold}°）")
    
    else:
        raise ValueError(f"Unknown weight strategy: {weight_strategy}")
    
    # 归一化（保持总权重=样本数）
    weights = weights / weights.mean()
    
    return weights


# ============================================================================
# 数据加载
# ============================================================================
def load_optimized_data(json_path):
    """加载优化后的数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"📦 加载数据: {len(data)}个样本")
    
    # 提取特征和标签
    features_list = []
    pid_list = []
    errors_list = []
    types = []
    
    for sample in data:
        # 使用简化的4维特征
        features = sample['features']
        feature_vec = [
            features['dof'],
            features['total_mass'],
            features['max_reach'],
            features['payload_mass']
        ]
        
        pid = sample['optimal_pid']
        pid_vec = [pid['kp'], pid['ki'], pid['kd']]
        
        # 获取优化误差（虚拟样本）或0（真实样本）
        error = sample.get('optimization_error_deg', 0.0)
        
        features_list.append(feature_vec)
        pid_list.append(pid_vec)
        errors_list.append(error)
        types.append(sample['type'])
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(pid_list, dtype=np.float32)
    errors = np.array(errors_list, dtype=np.float32)
    
    print(f"   特征形状: {X.shape}")
    print(f"   标签形状: {y.shape}")
    
    return X, y, errors, types, data


def normalize_data(X_train, X_test, y_train, y_test):
    """标准化数据"""
    # 特征标准化
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # PID标准化（log scale）
    y_train_log = np.log(y_train + 1e-8)
    y_test_log = np.log(y_test + 1e-8)
    
    y_mean = y_train_log.mean(axis=0)
    y_std = y_train_log.std(axis=0) + 1e-8
    y_train_norm = (y_train_log - y_mean) / y_std
    y_test_norm = (y_test_log - y_mean) / y_std
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, X_mean, X_std, y_mean, y_std


# ============================================================================
# 训练函数
# ============================================================================
def train_meta_pid_weighted(X_train, y_train, weights_train, X_val, y_val, weights_val, epochs=500, lr=1e-3):
    """训练加权元学习PID网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建模型
    model = SimplePIDPredictor(input_dim=4, hidden_dim=64, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedMSELoss()
    
    # 转换为Tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    weights_train_t = torch.FloatTensor(weights_train).to(device)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    weights_val_t = torch.FloatTensor(weights_val).to(device)
    
    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'weighted_val_loss': []}
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    print(f"\n🚀 开始加权训练... (epochs={epochs})")
    print(f"   训练样本权重范围: [{weights_train.min():.3f}, {weights_train.max():.3f}]")
    print(f"   训练样本平均权重: {weights_train.mean():.3f}")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t, weights_train_t)
        loss.backward()
        optimizer.step()
        
        # 验证（加权）
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss_weighted = criterion(val_pred, y_val_t, weights_val_t)
            # 也计算无权重损失用于监控
            val_loss_unweighted = ((val_pred - y_val_t) ** 2).mean()
        
        history['train_loss'].append(loss.item())
        history['weighted_val_loss'].append(val_loss_weighted.item())
        history['val_loss'].append(val_loss_unweighted.item())
        
        # Early stopping（基于加权验证损失）
        if val_loss_weighted < best_val_loss:
            best_val_loss = val_loss_weighted
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {loss.item():.6f}, "
                  f"Val Loss (weighted): {val_loss_weighted.item():.6f}, "
                  f"Val Loss (raw): {val_loss_unweighted.item():.6f}")
        
        if patience_counter >= patience:
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    
    print(f"✅ 训练完成！最佳加权验证损失: {best_val_loss:.6f}")
    
    return model, history


# ============================================================================
# 评估函数
# ============================================================================
def evaluate_weighted_model(model, X_test, y_test, errors_test, X_mean, X_std, y_mean, y_std):
    """评估加权模型"""
    device = next(model.parameters()).device
    
    # 标准化测试数据
    X_test_norm = (X_test - X_mean) / X_std
    y_test_log = np.log(y_test + 1e-8)
    y_test_norm = (y_test_log - y_mean) / y_std
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_norm).to(device)
        pred_norm = model(X_test_t).cpu().numpy()
    
    # 反标准化
    pred_log = pred_norm * y_std + y_mean
    pred = np.exp(pred_log)
    
    # 计算误差
    abs_errors = np.abs(pred - y_test)
    
    # 按优化误差分组评估
    low_error_mask = errors_test < 30
    high_error_mask = errors_test >= 30
    
    print(f"\n📊 评估结果:")
    print(f"\n全体样本 (n={len(X_test)}):")
    print(f"   Kp 绝对误差: {abs_errors[:, 0].mean():.4f}")
    print(f"   Ki 绝对误差: {abs_errors[:, 1].mean():.4f}")
    print(f"   Kd 绝对误差: {abs_errors[:, 2].mean():.4f}")
    print(f"   总体平均: {abs_errors.mean():.4f}")
    
    if low_error_mask.any():
        print(f"\n低优化误差样本 (优化误差<30°, n={low_error_mask.sum()}):")
        print(f"   Kp 绝对误差: {abs_errors[low_error_mask, 0].mean():.4f}")
        print(f"   总体平均: {abs_errors[low_error_mask].mean():.4f}")
    
    if high_error_mask.any():
        print(f"\n高优化误差样本 (优化误差≥30°, n={high_error_mask.sum()}):")
        print(f"   Kp 绝对误差: {abs_errors[high_error_mask, 0].mean():.4f}")
        print(f"   总体平均: {abs_errors[high_error_mask].mean():.4f}")
    
    return abs_errors, pred


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主训练流程"""
    print("=" * 80)
    print("加权元学习PID训练")
    print("=" * 80)
    
    # 1. 加载优化后的数据（过滤版：排除Laikago虚拟样本）
    data_path = Path(__file__).parent / 'augmented_pid_data_filtered.json'
    print(f"📁 加载数据: {data_path.name}")
    X_full, y_full, errors_full, types, data_full = load_optimized_data(data_path)
    
    # 2. 分析样本分布
    print(f"\n📊 样本优化误差统计:")
    print(f"   平均: {errors_full.mean():.2f}°")
    print(f"   中位: {np.median(errors_full):.2f}°")
    print(f"   最小: {errors_full.min():.2f}°")
    print(f"   最大: {errors_full.max():.2f}°")
    print(f"   <10°: {(errors_full < 10).sum()} 样本")
    print(f"   10-30°: {((errors_full >= 10) & (errors_full < 30)).sum()} 样本")
    print(f"   30-50°: {((errors_full >= 30) & (errors_full < 50)).sum()} 样本")
    print(f"   ≥50°: {(errors_full >= 50).sum()} 样本")
    
    # 3. 计算样本权重（测试三种策略）
    print(f"\n🔧 测试权重策略:")
    for strategy in ['inverse', 'exponential', 'threshold']:
        weights = compute_sample_weights(errors_full, strategy)
        print(f"\n   {strategy}:")
        print(f"      权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"      平均权重: {weights.mean():.3f}")
        print(f"      权重标准差: {weights.std():.3f}")
        
        # 显示不同误差段的权重
        low_err = errors_full < 30
        high_err = errors_full >= 50
        if low_err.any():
            print(f"      低误差(<30°)平均权重: {weights[low_err].mean():.3f}")
        if high_err.any():
            print(f"      高误差(≥50°)平均权重: {weights[high_err].mean():.3f}")
    
    # 4. 选择最佳策略并训练
    weight_strategy = 'strict'  # 使用strict策略（最高精度要求）
    
    print(f"\n✅ 选择权重策略: {weight_strategy}")
    print(f"   策略说明:")
    if weight_strategy == 'strict':
        print(f"      误差<25°: 权重1.0（保留）")
        print(f"      误差≥25°: 权重0.0（完全排除）")
        print(f"      目标：只用高质量样本，确保最高预测精度")
    elif weight_strategy == 'threshold':
        print(f"      误差<20°: 权重1.0（优秀）")
        print(f"      误差20-35°: 权重0.5（良好）")
        print(f"      误差≥35°: 权重0.05（差，基本忽略）")
    
    weights_full = compute_sample_weights(errors_full, weight_strategy)
    
    # 5. 划分训练/测试集
    X_train, X_test, y_train, y_test, weights_train, weights_test, errors_train, errors_test, idx_train, idx_test = train_test_split(
        X_full, y_full, weights_full, errors_full, np.arange(len(X_full)),
        test_size=0.2, random_state=42
    )
    
    print(f"\n📦 数据划分:")
    print(f"   训练样本: {len(X_train)}")
    print(f"   测试样本: {len(X_test)}")
    
    # 6. 标准化
    X_train_norm, X_test_norm, y_train_norm, y_test_norm, X_mean, X_std, y_mean, y_std = \
        normalize_data(X_train, X_test, y_train, y_test)
    
    # 7. 训练
    model, history = train_meta_pid_weighted(
        X_train_norm, y_train_norm, weights_train,
        X_test_norm, y_test_norm, weights_test,
        epochs=500, lr=1e-3
    )
    
    # 8. 评估
    test_data_subset = [data_full[i] for i in idx_test]
    abs_errors, pred = evaluate_weighted_model(
        model, X_test, y_test, errors_test,
        X_mean, X_std, y_mean, y_std
    )
    
    # 9. 保存模型
    model_save_path = Path(__file__).parent / 'meta_pid_weighted.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'weight_strategy': weight_strategy,
        'test_error_mean': abs_errors.mean(),
    }, model_save_path)
    print(f"\n💾 模型已保存: {model_save_path}")
    
    # 10. 可视化
    plt.figure(figsize=(12, 5))
    
    # 训练曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    plt.plot(history['weighted_val_loss'], label='Val Loss (Weighted)', alpha=0.8)
    plt.plot(history['val_loss'], label='Val Loss (Raw)', alpha=0.8, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve (Weighted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 误差分布
    plt.subplot(1, 2, 2)
    plt.scatter(errors_test, abs_errors.mean(axis=1), alpha=0.5, s=30)
    plt.xlabel('Optimization Error (degrees)')
    plt.ylabel('Prediction Error')
    plt.title('Prediction vs Optimization Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(__file__).parent / 'weighted_training_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 结果图已保存: {plot_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ 加权训练完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

