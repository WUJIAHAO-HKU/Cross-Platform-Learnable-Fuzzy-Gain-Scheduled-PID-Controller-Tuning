"""
鲁棒性测试脚本：测试PID vs RL+PID在扰动场景下的性能

假设：RL的优势可能在鲁棒性而非精度
测试场景：
1. 无扰动（基准）
2. 低强度扰动（1Nm随机力矩）
3. 中强度扰动（2Nm随机力矩）
4. 高强度扰动（3Nm随机力矩）
5. 负载变化（末端+1kg）
6. 模型不确定性（质量+30%）
"""

import yaml
import numpy as np
import argparse
import json
from stable_baselines3 import PPO
import pybullet as p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.franka_env import FrankaRLPIDEnv


class RobustFrankaEnv(FrankaRLPIDEnv):
    """
    增强版Franka环境，支持扰动和模型不确定性
    """
    def __init__(self, config, gui=False, disturbance_std=0.0, extra_mass=0.0, mass_uncertainty=0.0):
        """
        Args:
            config: 配置字典
            gui: 是否显示GUI
            disturbance_std: 扰动力矩标准差(Nm)
            extra_mass: 末端额外质量(kg)
            mass_uncertainty: 质量不确定性比例(0.3表示±30%)
        """
        self.disturbance_std = disturbance_std
        self.extra_mass = extra_mass
        self.mass_uncertainty = mass_uncertainty
        
        super().__init__(config, gui=gui)
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs, info = super().reset(seed=seed, options=options)
        
        # 添加末端负载
        if self.extra_mass > 0:
            # Franka的末端执行器是link 11
            p.changeDynamics(
                self.robot_id, 
                10,  # 末端关节
                mass=self.extra_mass,
                physicsClientId=self.client
            )
        
        # 添加质量不确定性
        if self.mass_uncertainty > 0:
            for i in range(7):  # 7个关节
                dyn_info = p.getDynamicsInfo(self.robot_id, i, physicsClientId=self.client)
                original_mass = dyn_info[0]
                # 随机偏差±mass_uncertainty
                mass_factor = 1.0 + np.random.uniform(-self.mass_uncertainty, self.mass_uncertainty)
                new_mass = original_mass * mass_factor
                p.changeDynamics(
                    self.robot_id,
                    i,
                    mass=new_mass,
                    physicsClientId=self.client
                )
        
        return obs, info
    
    def step(self, action):
        """执行一步，添加扰动支持"""
        # 获取当前状态
        q, qd = self._get_robot_state()
        t = self.current_step * self.dt
        qref, qd_ref = self.traj_gen.get_reference(t)
        
        # 计算控制力矩（PID + RL补偿）
        class TempPolicy:
            def __init__(self, action):
                self.action = action
            def predict(self, state, deterministic=True):
                return self.action, None
        
        self.controller.rl_policy = TempPolicy(action)
        tau_total, info = self.controller.compute_control(q, qd, qref, qd_ref, training=True)
        
        # ⭐ 添加扰动
        disturbance = np.zeros(7)
        if self.disturbance_std > 0:
            disturbance = np.random.normal(0, self.disturbance_std, size=7)
            tau_total = tau_total + disturbance
            # 重新限幅
            tau_max = self.config.get('robot_params', {}).get('tau_max', 87.0)
            tau_total = np.clip(tau_total, -tau_max, tau_max)
        
        # 应用力矩
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.TORQUE_CONTROL,
            forces=tau_total
        )
        p.stepSimulation()
        
        # 新状态
        q_new, qd_new = self._get_robot_state()
        qref_new, qd_ref_new = self.traj_gen.get_reference(t + self.dt)
        next_state = self.controller._construct_state(q_new, qd_new, qref_new)
        
        # 导入compute_reward
        from controllers.rl_pid_hybrid import compute_reward
        
        # 计算奖励
        reward, reward_info = compute_reward(
            q_new, qd_new, qref_new, action, info['delta_tau'], 
            self.config.get('rl_params', {})
        )
        
        # 检查终止
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # 安全检查：发散检测
        if np.any(np.abs(q_new) > 3.0) or np.any(np.isnan(q_new)):
            reward -= 1000
            terminated = True
        
        # 合并info，添加扰动信息
        step_info = {
            **info,
            **reward_info,
            'tracking_error': np.linalg.norm(qref_new - q_new),
            'q': q_new,
            'qref': qref_new,
            'disturbance': disturbance  # 新增
        }
        
        return next_state, reward, terminated, truncated, step_info


# 定义测试场景
ROBUSTNESS_SCENARIOS = [
    {
        'name': '无扰动',
        'disturbance_std': 0.0,
        'extra_mass': 0.0,
        'mass_uncertainty': 0.0,
        'description': '基准场景，无任何扰动'
    },
    {
        'name': '低强度扰动',
        'disturbance_std': 1.0,
        'extra_mass': 0.0,
        'mass_uncertainty': 0.0,
        'description': '1Nm随机力矩扰动'
    },
    {
        'name': '中强度扰动',
        'disturbance_std': 2.0,
        'extra_mass': 0.0,
        'mass_uncertainty': 0.0,
        'description': '2Nm随机力矩扰动'
    },
    {
        'name': '高强度扰动',
        'disturbance_std': 3.0,
        'extra_mass': 0.0,
        'mass_uncertainty': 0.0,
        'description': '3Nm随机力矩扰动'
    },
    {
        'name': '末端负载',
        'disturbance_std': 0.0,
        'extra_mass': 1.0,
        'mass_uncertainty': 0.0,
        'description': '末端增加1kg负载'
    },
    {
        'name': '模型不确定性',
        'disturbance_std': 0.0,
        'extra_mass': 0.0,
        'mass_uncertainty': 0.3,
        'description': '关节质量±30%随机偏差'
    }
]


def run_robust_test(model_path, config_path, scenario, num_steps=5000, use_rl=False):
    """
    运行鲁棒性测试
    
    Args:
        model_path: RL模型路径
        config_path: 配置文件路径
        scenario: 场景配置
        num_steps: 测试步数
        use_rl: 是否使用RL
    
    Returns:
        dict: 测试结果
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建环境（如果use_rl需要加载模型）
    env = RobustFrankaEnv(
        config,
        gui=False,
        disturbance_std=scenario['disturbance_std'],
        extra_mass=scenario['extra_mass'],
        mass_uncertainty=scenario['mass_uncertainty']
    )
    
    if use_rl:
        model = PPO.load(model_path.replace('.zip', ''))
    
    obs, _ = env.reset()
    
    errors = []
    rewards = []
    delta_taus = []
    disturbances = []
    
    for step in range(num_steps):
        if use_rl:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)
        
        obs, reward, done, truncated, info = env.step(action)
        
        # 获取跟踪误差
        if 'tracking_error' in info:
            err_norm = info['tracking_error']
        else:
            # 计算误差范数
            q_err = info['q'] - info['qref']
            err_norm = np.linalg.norm(q_err)
        
        errors.append(err_norm)
        rewards.append(reward)
        
        if use_rl and 'delta_tau' in info:
            delta_taus.append(np.linalg.norm(info['delta_tau']))
        
        if 'disturbance' in info:
            disturbances.append(np.linalg.norm(info['disturbance']))
        
        if done or truncated:
            break
    
    env.close()
    
    # 统计
    errors = np.array(errors)
    results = {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors)),
        'total_reward': float(np.sum(rewards)),
    }
    
    if use_rl and delta_taus:
        results['mean_delta_tau'] = float(np.mean(delta_taus))
    
    if disturbances:
        results['mean_disturbance'] = float(np.mean(disturbances))
    
    return results


def test_all_robustness_scenarios(model_path, config_path, num_repeats=3, num_steps=5000):
    """
    测试所有鲁棒性场景
    
    Args:
        model_path: RL模型路径
        config_path: 配置文件路径
        num_repeats: 每个场景重复次数
        num_steps: 每次测试步数
    
    Returns:
        dict: 所有测试结果
    """
    print("=" * 80)
    print("鲁棒性测试")
    print("=" * 80)
    print(f"模型: {model_path}")
    print(f"配置: {config_path}")
    print(f"重复次数: {num_repeats}")
    print(f"测试步数: {num_steps}")
    print("=" * 80)
    
    all_results = {}
    
    for scenario in ROBUSTNESS_SCENARIOS:
        print(f"\n测试场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        print("-" * 80)
        
        scenario_results = {
            'name': scenario['name'],
            'description': scenario['description'],
            'params': {k: v for k, v in scenario.items() if k not in ['name', 'description']},
            'pid_results': [],
            'rl_results': []
        }
        
        for repeat in range(num_repeats):
            print(f"  重复 {repeat+1}/{num_repeats}...")
            
            # 测试纯PID
            print("    [1/2] 纯PID测试中...")
            pid_result = run_robust_test(model_path, config_path, scenario, num_steps, use_rl=False)
            scenario_results['pid_results'].append(pid_result)
            print(f"          平均误差: {pid_result['mean_error']:.4f}弧度 ({np.rad2deg(pid_result['mean_error']):.2f}度)")
            
            # 测试RL+PID
            print("    [2/2] RL+PID测试中...")
            rl_result = run_robust_test(model_path, config_path, scenario, num_steps, use_rl=True)
            scenario_results['rl_results'].append(rl_result)
            print(f"          平均误差: {rl_result['mean_error']:.4f}弧度 ({np.rad2deg(rl_result['mean_error']):.2f}度)")
        
        # 计算平均
        pid_mean = np.mean([r['mean_error'] for r in scenario_results['pid_results']])
        rl_mean = np.mean([r['mean_error'] for r in scenario_results['rl_results']])
        improvement = (pid_mean - rl_mean) / pid_mean * 100
        
        scenario_results['summary'] = {
            'pid_mean_error': float(pid_mean),
            'pid_mean_error_deg': float(np.rad2deg(pid_mean)),
            'rl_mean_error': float(rl_mean),
            'rl_mean_error_deg': float(np.rad2deg(rl_mean)),
            'improvement_percent': float(improvement)
        }
        
        print(f"\n  📊 场景总结:")
        print(f"     纯PID:  {np.rad2deg(pid_mean):.2f}度")
        print(f"     RL+PID: {np.rad2deg(rl_mean):.2f}度")
        print(f"     改进率: {improvement:+.2f}%")
        
        all_results[scenario['name']] = scenario_results
    
    return all_results


def generate_robustness_plots(results, output_dir='results/robustness'):
    """生成鲁棒性对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = list(results.keys())
    pid_errors = [results[s]['summary']['pid_mean_error_deg'] for s in scenarios]
    rl_errors = [results[s]['summary']['rl_mean_error_deg'] for s in scenarios]
    improvements = [results[s]['summary']['improvement_percent'] for s in scenarios]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 误差对比
    ax1 = axes[0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, pid_errors, width, label='Pure PID', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, rl_errors, width, label='RL+PID', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Tracking Error (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Robustness Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=20, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 改进率
    ax2 = axes[1]
    colors = ['green' if i > 10 else 'orange' if i > 5 else 'lightcoral' for i in improvements]
    bars = ax2.bar(scenarios, improvements, color=colors, alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('RL Improvement Rate', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(scenarios, rotation=20, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(y=10, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    
    # 3. 扰动强度vs改进率
    ax3 = axes[2]
    disturbance_scenarios = [s for s in scenarios if '扰动' in s or '无扰动' in s]
    disturbance_levels = [0, 1, 2, 3]  # 对应无、低、中、高
    disturbance_improvements = [results[s]['summary']['improvement_percent'] 
                               for s in disturbance_scenarios]
    
    ax3.plot(disturbance_levels, disturbance_improvements, 'o-', 
            linewidth=2, markersize=10, color='darkblue', label='RL Improvement')
    ax3.fill_between(disturbance_levels, 0, disturbance_improvements, alpha=0.2, color='blue')
    
    ax3.set_xlabel('Disturbance Level (Nm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RL Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Improvement vs Disturbance Intensity', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(y=10, color='green', linestyle=':', linewidth=1, alpha=0.5, label='10% Threshold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/robustness_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 鲁棒性对比图已保存: {output_dir}/robustness_comparison.png")


def generate_robustness_report(results, output_dir='results/robustness'):
    """生成鲁棒性测试报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'{output_dir}/robustness_report_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("鲁棒性测试报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总结
        all_improvements = [results[s]['summary']['improvement_percent'] for s in results]
        avg_improvement = np.mean(all_improvements)
        max_improvement = max(all_improvements)
        max_scenario = max(results.keys(), key=lambda s: results[s]['summary']['improvement_percent'])
        
        f.write(f"平均改进率: {avg_improvement:.2f}%\n")
        f.write(f"最大改进率: {max_improvement:.2f}% ({max_scenario})\n")
        f.write(f"显著改进场景 (>10%): {sum(1 for i in all_improvements if i > 10)}个\n\n")
        
        # 各场景详细结果
        f.write("【各场景详细结果】\n")
        f.write("=" * 80 + "\n\n")
        
        for scenario_name, data in results.items():
            f.write(f"场景: {scenario_name}\n")
            f.write(f"描述: {data['description']}\n")
            f.write(f"参数: {data['params']}\n")
            f.write("-" * 80 + "\n")
            
            summary = data['summary']
            f.write(f"纯PID误差: {summary['pid_mean_error_deg']:.2f}度\n")
            f.write(f"RL+PID误差: {summary['rl_mean_error_deg']:.2f}度\n")
            f.write(f"改进率: {summary['improvement_percent']:+.2f}%\n\n")
        
        # 结论
        f.write("【结论】\n")
        f.write("=" * 80 + "\n")
        
        significant_scenarios = [s for s in results if results[s]['summary']['improvement_percent'] > 10]
        
        if len(significant_scenarios) >= 2:
            f.write("✅ RL在多个扰动场景下显示出显著优势！\n")
            f.write("   建议：论文可以强调RL的鲁棒性优势\n")
        elif len(significant_scenarios) == 1:
            f.write("⚠️  RL仅在特定场景下有显著改进\n")
            f.write("   建议：讨论RL的适用场景和局限性\n")
        else:
            f.write("❌ RL即使在鲁棒性测试中也没有显著优势\n")
            f.write("   建议：接受PID优化已足够好的结论，或重新设计RL训练策略\n")
    
    print(f"✅ 鲁棒性测试报告已保存: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='鲁棒性测试')
    parser.add_argument('--model', type=str, default='logs/best_model/best_model.zip')
    parser.add_argument('--config', type=str, default='configs/stage1_optimized.yaml')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--output', type=str, default='results/robustness')
    args = parser.parse_args()
    
    # 运行测试
    results = test_all_robustness_scenarios(args.model, args.config, args.repeats, args.steps)
    
    # 保存原始数据
    os.makedirs(args.output, exist_ok=True)
    json_path = f'{args.output}/raw_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 原始数据已保存: {json_path}")
    
    # 生成图表
    generate_robustness_plots(results, args.output)
    
    # 生成报告
    generate_robustness_report(results, args.output)
    
    print("\n" + "=" * 80)
    print("✅ 鲁棒性测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

