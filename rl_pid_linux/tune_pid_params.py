"""
🔧 PID参数自动优化脚本

目标：找到使跟踪误差最小且稳定的PID参数

评估标准：
1. 跟踪误差（主要）
2. 控制力矩平滑度（避免震荡）
3. 稳定性（避免发散）
"""

import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json

def evaluate_pid_params(Kp_scale, Ki_scale, Kd_scale, config_template, n_steps=2000, verbose=False):
    """
    评估一组PID参数的性能
    
    参数：
        Kp_scale, Ki_scale, Kd_scale: 相对于基准值的缩放因子
        config_template: 配置模板
        n_steps: 评估步数
    
    返回：
        score: 综合评分（越小越好）
        metrics: 详细指标
    """
    # 基准PID参数（经验值）
    Kp_base = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 30.0, 20.0])
    Ki_base = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.1, 0.1])
    Kd_base = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 3.0, 2.0])
    
    # 应用缩放因子
    Kp = Kp_base * Kp_scale
    Ki = Ki_base * Ki_scale
    Kd = Kd_base * Kd_scale
    
    # 创建配置
    config = config_template.copy()
    config['pid_params']['Kp'] = Kp.tolist()
    config['pid_params']['Ki'] = Ki.tolist()
    config['pid_params']['Kd'] = Kd.tolist()
    config['pid_params']['enable_gravity_compensation'] = False
    
    # 创建环境
    try:
        env = FrankaRLPIDEnv(config, gui=False)
    except Exception as e:
        if verbose:
            print(f"  ❌ 环境创建失败: {e}")
        return 1e6, None  # 返回极大惩罚
    
    # 运行仿真
    obs, _ = env.reset()
    errors = []
    tau_pid_list = []
    
    try:
        for step in range(n_steps):
            # 纯PID控制（action=0）
            action = np.zeros(7, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            errors.append(info['err_norm'])
            tau_pid_list.append(info.get('tau_pid', np.zeros(7)))
            
            # 检查稳定性
            if info['err_norm'] > 3.0:  # 误差>3弧度认为不稳定
                if verbose:
                    print(f"  ⚠️ 第{step}步不稳定，误差={info['err_norm']:.2f}")
                env.close()
                return 1e6, None  # 不稳定，返回极大惩罚
                
    except Exception as e:
        if verbose:
            print(f"  ❌ 仿真失败: {e}")
        env.close()
        return 1e6, None
    
    env.close()
    
    # 计算评估指标
    errors = np.array(errors)
    tau_pid_array = np.array(tau_pid_list)
    
    # 跳过前100步（预热期）
    errors_stable = errors[100:]
    tau_pid_stable = tau_pid_array[100:]
    
    # 1. 跟踪误差（主要指标，权重最大）
    mean_error = np.mean(errors_stable)
    max_error = np.max(errors_stable)
    std_error = np.std(errors_stable)
    
    # 2. 控制力矩平滑度（避免震荡）
    tau_diff = np.diff(tau_pid_stable, axis=0)
    tau_smoothness = np.mean(np.abs(tau_diff))  # 力矩变化率
    
    # 3. 控制力矩幅值
    tau_magnitude = np.mean(np.abs(tau_pid_stable))
    
    # 综合评分（加权和）
    score = (
        10.0 * mean_error +      # 平均误差（最重要）
        2.0 * max_error +         # 最大误差
        1.0 * std_error +         # 误差波动
        0.01 * tau_smoothness +   # 力矩平滑度
        0.001 * tau_magnitude     # 力矩幅值
    )
    
    metrics = {
        'mean_error': float(mean_error),
        'max_error': float(max_error),
        'std_error': float(std_error),
        'tau_smoothness': float(tau_smoothness),
        'tau_magnitude': float(tau_magnitude),
        'score': float(score),
        'Kp': [float(x) for x in Kp],
        'Ki': [float(x) for x in Ki],
        'Kd': [float(x) for x in Kd],
        'Kp_scale': float(Kp_scale),
        'Ki_scale': float(Ki_scale),
        'Kd_scale': float(Kd_scale)
    }
    
    if verbose:
        print(f"  误差: {mean_error:.4f}±{std_error:.4f} (最大{max_error:.4f})")
        print(f"  平滑度: {tau_smoothness:.2f}, 力矩: {tau_magnitude:.2f}")
        print(f"  综合得分: {score:.2f}")
    
    return score, metrics


def grid_search(config_template, n_steps=2000):
    """网格搜索PID参数"""
    print("=" * 70)
    print("🔍 方法1: 网格搜索")
    print("=" * 70)
    
    # 搜索空间（相对于基准值的缩放因子）
    Kp_scales = [0.5, 1.0, 2.0, 3.0, 4.0]
    Ki_scales = [0.1, 0.5, 1.0, 2.0]
    Kd_scales = [0.5, 1.0, 2.0, 3.0]
    
    total_tests = len(Kp_scales) * len(Ki_scales) * len(Kd_scales)
    print(f"搜索空间: {total_tests}组参数")
    print()
    
    best_score = float('inf')
    best_metrics = None
    all_results = []
    
    with tqdm(total=total_tests, desc="网格搜索进度") as pbar:
        for Kp_s in Kp_scales:
            for Ki_s in Ki_scales:
                for Kd_s in Kd_scales:
                    score, metrics = evaluate_pid_params(
                        Kp_s, Ki_s, Kd_s, 
                        config_template, 
                        n_steps=n_steps,
                        verbose=False
                    )
                    
                    if metrics is not None:
                        all_results.append(metrics)
                        
                        if score < best_score:
                            best_score = score
                            best_metrics = metrics
                            tqdm.write(f"✨ 新最优: Kp×{Kp_s:.1f}, Ki×{Ki_s:.1f}, Kd×{Kd_s:.1f} "
                                     f"→ 误差={metrics['mean_error']:.4f}, 得分={score:.2f}")
                    
                    pbar.update(1)
    
    return best_metrics, all_results


def bayesian_optimization(config_template, n_iterations=50, n_steps=2000):
    """贝叶斯优化（使用简单的随机搜索 + 局部优化）"""
    print("\n" + "=" * 70)
    print("🎯 方法2: 贝叶斯优化（局部精细搜索）")
    print("=" * 70)
    
    # 从网格搜索的最优结果附近开始
    print("基于网格搜索结果，在最优点附近进行精细搜索...")
    print()
    
    def objective(x):
        """优化目标函数"""
        Kp_s, Ki_s, Kd_s = x
        # 参数约束
        if Kp_s < 0.1 or Kp_s > 10.0:
            return 1e6
        if Ki_s < 0.01 or Ki_s > 5.0:
            return 1e6
        if Kd_s < 0.1 or Kd_s > 10.0:
            return 1e6
        
        score, _ = evaluate_pid_params(Kp_s, Ki_s, Kd_s, config_template, n_steps=n_steps)
        return score
    
    # 使用Nelder-Mead算法（不需要梯度）
    from scipy.optimize import minimize
    
    # 多次随机初始化，选择最好的
    best_result = None
    best_score_global = float('inf')
    
    # 候选初始点（基于经验）
    initial_points = [
        [2.0, 0.5, 2.0],   # 中等增益
        [3.0, 1.0, 2.0],   # 高Kp
        [4.0, 0.5, 3.0],   # 很高Kp
        [2.0, 0.1, 1.0],   # 低Ki
        [3.0, 0.5, 1.5],   # 平衡型
    ]
    
    for i, x0 in enumerate(initial_points):
        print(f"\n优化尝试 {i+1}/{len(initial_points)}: 初始点 Kp×{x0[0]}, Ki×{x0[1]}, Kd×{x0[2]}")
        
        result = minimize(
            objective, 
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 30, 'disp': False}
        )
        
        if result.fun < best_score_global:
            best_score_global = result.fun
            best_result = result
            print(f"  ✨ 新最优得分: {result.fun:.2f}")
    
    # 获取最优参数的详细指标
    Kp_opt, Ki_opt, Kd_opt = best_result.x
    _, best_metrics = evaluate_pid_params(
        Kp_opt, Ki_opt, Kd_opt, 
        config_template, 
        n_steps=n_steps,
        verbose=True
    )
    
    return best_metrics


def main():
    """主函数"""
    print("=" * 70)
    print("🔧 PID参数自动优化")
    print("=" * 70)
    print()
    
    # 加载配置模板
    with open('configs/stage1_final.yaml', 'r') as f:
        config_template = yaml.safe_load(f)
    
    print(f"轨迹类型: {config_template['trajectory']['type']}")
    print(f"轨迹速度: {config_template['trajectory']['speed']} rad/s")
    print(f"轨迹幅度: {config_template['trajectory']['amplitude']} rad")
    print()
    
    # 阶段1: 网格搜索（粗略搜索）
    best_grid, all_results = grid_search(config_template, n_steps=1000)
    
    print("\n" + "=" * 70)
    print("📊 网格搜索最优结果")
    print("=" * 70)
    print(f"Kp缩放: {best_grid['Kp_scale']:.2f}")
    print(f"Ki缩放: {best_grid['Ki_scale']:.2f}")
    print(f"Kd缩放: {best_grid['Kd_scale']:.2f}")
    print(f"\n实际PID参数:")
    print(f"Kp: {np.array(best_grid['Kp'])}")
    print(f"Ki: {np.array(best_grid['Ki'])}")
    print(f"Kd: {np.array(best_grid['Kd'])}")
    print(f"\n性能指标:")
    print(f"平均误差: {best_grid['mean_error']:.4f} 弧度 ({np.rad2deg(best_grid['mean_error']):.2f}度)")
    print(f"最大误差: {best_grid['max_error']:.4f} 弧度 ({np.rad2deg(best_grid['max_error']):.2f}度)")
    print(f"误差标准差: {best_grid['std_error']:.4f}")
    
    # 阶段2: 局部精细搜索
    # 以网格搜索最优点为中心
    config_template['pid_params']['Kp'] = best_grid['Kp']
    config_template['pid_params']['Ki'] = best_grid['Ki']
    config_template['pid_params']['Kd'] = best_grid['Kd']
    
    best_bayes = bayesian_optimization(config_template, n_iterations=50, n_steps=2000)
    
    print("\n" + "=" * 70)
    print("🏆 最终优化结果")
    print("=" * 70)
    print(f"\n最优PID参数:")
    print(f"Kp: {np.array(best_bayes['Kp'])}")
    print(f"Ki: {np.array(best_bayes['Ki'])}")
    print(f"Kd: {np.array(best_bayes['Kd'])}")
    print(f"\n性能指标:")
    print(f"平均误差: {best_bayes['mean_error']:.4f} 弧度 ({np.rad2deg(best_bayes['mean_error']):.2f}度)")
    print(f"最大误差: {best_bayes['max_error']:.4f} 弧度 ({np.rad2deg(best_bayes['max_error']):.2f}度)")
    print(f"误差标准差: {best_bayes['std_error']:.4f}")
    print(f"综合得分: {best_bayes['score']:.2f}")
    
    # 保存结果
    output_file = 'pid_tuning_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'best_params': best_bayes,
            'grid_search_best': best_grid,
            'all_grid_results': all_results
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 可视化对比
    print("\n生成对比图...")
    visualize_results(all_results, best_bayes)
    print("✅ 图表已保存到: pid_tuning_visualization.png")
    
    # 生成配置文件
    print("\n生成优化后的配置文件...")
    config_optimized = config_template.copy()
    config_optimized['pid_params']['Kp'] = best_bayes['Kp']
    config_optimized['pid_params']['Ki'] = best_bayes['Ki']
    config_optimized['pid_params']['Kd'] = best_bayes['Kd']
    
    with open('configs/stage1_optimized.yaml', 'w') as f:
        yaml.dump(config_optimized, f, default_flow_style=False, sort_keys=False)
    
    print("✅ 优化配置已保存到: configs/stage1_optimized.yaml")
    
    print("\n" + "=" * 70)
    print("🎯 下一步：使用优化后的PID参数训练RL")
    print("=" * 70)
    print("python training/train_ppo.py --config configs/stage1_optimized.yaml")


def visualize_results(all_results, best_result):
    """可视化调参结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取数据
    mean_errors = [r['mean_error'] for r in all_results]
    max_errors = [r['max_error'] for r in all_results]
    Kp_scales = [r['Kp_scale'] for r in all_results]
    Ki_scales = [r['Ki_scale'] for r in all_results]
    Kd_scales = [r['Kd_scale'] for r in all_results]
    scores = [r['score'] for r in all_results]
    
    # 子图1: 误差 vs Kp缩放
    axes[0, 0].scatter(Kp_scales, mean_errors, alpha=0.6, s=50, c=scores, cmap='RdYlGn_r')
    axes[0, 0].scatter(best_result['Kp_scale'], best_result['mean_error'], 
                      color='red', s=200, marker='*', label='最优', zorder=5)
    axes[0, 0].set_xlabel('Kp缩放因子', fontsize=12)
    axes[0, 0].set_ylabel('平均误差 (弧度)', fontsize=12)
    axes[0, 0].set_title('误差 vs Kp缩放', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 误差 vs Ki缩放
    axes[0, 1].scatter(Ki_scales, mean_errors, alpha=0.6, s=50, c=scores, cmap='RdYlGn_r')
    axes[0, 1].scatter(best_result['Ki_scale'], best_result['mean_error'], 
                      color='red', s=200, marker='*', label='最优', zorder=5)
    axes[0, 1].set_xlabel('Ki缩放因子', fontsize=12)
    axes[0, 1].set_ylabel('平均误差 (弧度)', fontsize=12)
    axes[0, 1].set_title('误差 vs Ki缩放', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 误差 vs Kd缩放
    axes[1, 0].scatter(Kd_scales, mean_errors, alpha=0.6, s=50, c=scores, cmap='RdYlGn_r')
    axes[1, 0].scatter(best_result['Kd_scale'], best_result['mean_error'], 
                      color='red', s=200, marker='*', label='最优', zorder=5)
    axes[1, 0].set_xlabel('Kd缩放因子', fontsize=12)
    axes[1, 0].set_ylabel('平均误差 (弧度)', fontsize=12)
    axes[1, 0].set_title('误差 vs Kd缩放', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 综合得分分布
    axes[1, 1].hist(scores, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(best_result['score'], color='red', linestyle='--', 
                      linewidth=2, label=f'最优得分: {best_result["score"]:.2f}')
    axes[1, 1].set_xlabel('综合得分', fontsize=12)
    axes[1, 1].set_ylabel('频数', fontsize=12)
    axes[1, 1].set_title('得分分布', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pid_tuning_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()

