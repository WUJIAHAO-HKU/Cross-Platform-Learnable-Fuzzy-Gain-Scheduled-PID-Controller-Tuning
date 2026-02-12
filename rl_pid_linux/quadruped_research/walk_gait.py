#!/usr/bin/env python3
"""
Walk步态控制器（四拍步态）
目标：稳定、慢速行走，最高稳定性
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入基础Laikago控制器
import sys
sys.path.append(str(Path(__file__).parent))
from test_laikago_final import LaikagoRobot


class WalkGaitController:
    """Walk步态控制器 - 四拍步态"""
    
    # 腿部顺序映射
    LEG_ORDER = ['FR', 'FL', 'RR', 'RL']  # 前右 → 前左 → 后右 → 后左
    LEG_TO_INDEX = {'FR': 0, 'FL': 1, 'RR': 2, 'RL': 3}
    
    def __init__(self, robot: LaikagoRobot):
        """
        初始化
        
        Args:
            robot: LaikagoRobot实例
        """
        self.robot = robot
        
        # Walk步态参数
        self.frequency = 0.5          # Hz（比Trot慢）
        self.duty_cycle = 0.75        # 75%时间支撑
        self.step_height = 0.08       # 抬腿高度（米）
        self.stride_length = 0.05     # 步长（米，较小）
        
        # 站立姿态（基准）
        self.stance_hip_angle = 1.0
        self.stance_knee_angle = -2.0
        self.stance_abd_angle = 0.0   # abduction
        
        # 摆动腿参数
        self.swing_hip_forward = 0.7   # 摆动时大腿前伸
        self.swing_hip_backward = 1.2  # 摆动后大腿后伸
        
        # 数据记录
        self.time_log = []
        self.base_pos_log = []
        self.phase_log = []
        self.swing_leg_log = []
        
    def get_swing_leg(self, phase):
        """
        根据相位确定当前摆动的腿
        
        Args:
            phase: 0-1之间的相位
        
        Returns:
            leg_name: 'FR', 'FL', 'RR', 'RL' 或 None
            leg_phase: 该腿的局部相位 (0-1)
        """
        # 将相位分成4段，每段25%
        if phase < 0.25:
            return 'FR', phase / 0.25
        elif phase < 0.5:
            return 'FL', (phase - 0.25) / 0.25
        elif phase < 0.75:
            return 'RR', (phase - 0.5) / 0.25
        else:
            return 'RL', (phase - 0.75) / 0.25
    
    def compute_swing_trajectory(self, leg_phase):
        """
        计算摆动腿的关节角度（抛物线轨迹）
        
        Args:
            leg_phase: 0-1，该腿的局部相位
        
        Returns:
            hip_angle: 大腿角度
            knee_angle: 小腿角度（保持固定）
        """
        # 抛物线轨迹：前半程抬腿，后半程落腿
        if leg_phase < 0.5:
            # 抬腿阶段：从后向前，同时抬高
            t = leg_phase * 2  # 0-1
            hip_angle = self.swing_hip_backward + \
                       (self.swing_hip_forward - self.swing_hip_backward) * t
        else:
            # 落腿阶段：从前向后，同时降低
            t = (leg_phase - 0.5) * 2  # 0-1
            hip_angle = self.swing_hip_forward + \
                       (self.swing_hip_backward - self.swing_hip_forward) * t
        
        # 小腿保持固定角度
        knee_angle = self.stance_knee_angle
        
        return hip_angle, knee_angle
    
    def generate_walk_action(self, t):
        """
        生成Walk步态动作
        
        Args:
            t: 当前时间（秒）
        
        Returns:
            action: 关节角度 (12,)
            phase: 当前步态相位 (0-1)
            swing_leg: 当前摆动的腿
        """
        # 计算步态相位
        phase = (t * self.frequency) % 1.0
        
        # 确定摆动腿
        swing_leg, leg_phase = self.get_swing_leg(phase)
        
        # 初始化所有关节为站立姿态
        action = np.array([
            self.stance_abd_angle, self.stance_hip_angle, self.stance_knee_angle,  # FR
            self.stance_abd_angle, self.stance_hip_angle, self.stance_knee_angle,  # FL
            self.stance_abd_angle, self.stance_hip_angle, self.stance_knee_angle,  # RR
            self.stance_abd_angle, self.stance_hip_angle, self.stance_knee_angle   # RL
        ])
        
        # 修改摆动腿的角度
        if swing_leg:
            leg_idx = self.LEG_TO_INDEX[swing_leg]
            swing_hip, swing_knee = self.compute_swing_trajectory(leg_phase)
            
            # 更新该腿的关节角度
            action[leg_idx * 3 + 1] = swing_hip    # hip
            action[leg_idx * 3 + 2] = swing_knee   # knee
        
        # 记录数据
        self.time_log.append(t)
        state = self.robot.get_state()
        self.base_pos_log.append(state['base_pos'].copy())
        self.phase_log.append(phase)
        self.swing_leg_log.append(swing_leg if swing_leg else 'None')
        
        return action, phase, swing_leg
    
    def run_walk(self, duration=10.0, verbose=True):
        """
        运行Walk步态
        
        Args:
            duration: 持续时间（秒）
            verbose: 是否打印信息
        
        Returns:
            results: 包含性能指标的字典
        """
        # 重置机器人
        self.robot.reset()
        
        # 清空日志
        self.time_log = []
        self.base_pos_log = []
        self.phase_log = []
        self.swing_leg_log = []
        
        if verbose:
            print(f"\n🚶 开始Walk步态（持续{duration}秒）")
            print(f"   频率: {self.frequency} Hz")
            print(f"   Duty cycle: {self.duty_cycle} (75%支撑)")
            print(f"   步高: {self.step_height}m")
        
        t = 0
        dt = 0.001
        steps = int(duration / dt)
        
        for i in range(steps):
            # 生成动作
            action, phase, swing_leg = self.generate_walk_action(t)
            
            # 应用动作
            self.robot.apply_action(action, motor_kp=0.5, motor_kd=0.1)
            p.stepSimulation(physicsClientId=self.robot.client)
            time.sleep(dt)
            t += dt
            
            # 每2秒打印状态
            if verbose and i % 2000 == 0:
                state = self.robot.get_state()
                height = state['base_pos'][2]
                pos_x = state['base_pos'][0]
                pos_y = state['base_pos'][1]
                print(f"   t={t:.1f}s: 高度={height:.3f}m, "
                      f"X={pos_x:.3f}m, Y={pos_y:.3f}m, "
                      f"摆动腿={swing_leg}, 相位={phase:.2f}")
        
        # 最终评估
        final_state = self.robot.get_state()
        height = final_state['base_pos'][2]
        distance_x = final_state['base_pos'][0]
        distance_y = abs(final_state['base_pos'][1])
        
        # 计算平均横向偏移
        pos_array = np.array(self.base_pos_log)
        avg_lateral_drift = np.mean(np.abs(pos_array[:, 1]))
        max_lateral_drift = np.max(np.abs(pos_array[:, 1]))
        
        # 计算高度稳定性
        avg_height = np.mean(pos_array[:, 2])
        height_std = np.std(pos_array[:, 2])
        
        results = {
            'final_height': height,
            'avg_height': avg_height,
            'height_std': height_std,
            'distance_forward': distance_x,
            'distance_lateral': distance_y,
            'avg_lateral_drift': avg_lateral_drift,
            'max_lateral_drift': max_lateral_drift,
            'forward_speed': distance_x / duration,
            'duration': duration
        }
        
        if verbose:
            print(f"\n📊 Walk步态结果:")
            print(f"   前进距离: {distance_x:.3f}m")
            print(f"   横向偏移: {distance_y:.3f}m")
            print(f"   平均横向漂移: {avg_lateral_drift:.3f}m")
            print(f"   前进速度: {results['forward_speed']:.3f}m/s")
            print(f"   平均高度: {avg_height:.3f}m ± {height_std:.3f}m")
        
        return results
    
    def plot_analysis(self, save_path=None):
        """
        绘制分析图表
        
        Args:
            save_path: 保存路径（可选）
        """
        if len(self.time_log) == 0:
            print("⚠️  没有数据可绘制")
            return
        
        time_array = np.array(self.time_log)
        pos_array = np.array(self.base_pos_log)
        phase_array = np.array(self.phase_log)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 14))
        
        # 子图1: XY轨迹
        axes[0].plot(pos_array[:, 0], pos_array[:, 1], 'b-', linewidth=2, label='Base轨迹')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Y=0')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('Walk Gait - Base Trajectory (Top View)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # 子图2: 横向偏移
        axes[1].plot(time_array, pos_array[:, 1], 'b-', linewidth=2, label='Lateral Drift')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].fill_between(time_array, -0.3, 0.3, alpha=0.1, color='green', label='Target ±0.3m')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_title('Lateral Drift Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 子图3: 高度变化
        axes[2].plot(time_array, pos_array[:, 2], 'g-', linewidth=2, label='Height')
        axes[2].axhline(y=0.204, color='orange', linestyle='--', alpha=0.5, label='Target 0.204m')
        axes[2].fill_between(time_array, 0.18, 0.25, alpha=0.1, color='green')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Height (m)')
        axes[2].set_title('Height Stability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 子图4: 步态相位
        axes[3].plot(time_array, phase_array, 'purple', linewidth=1, label='Phase')
        # 标记各腿摆动区间
        colors = {'FR': 'red', 'FL': 'blue', 'RR': 'green', 'RL': 'orange'}
        for i, (t, phase, leg) in enumerate(zip(time_array[::100], phase_array[::100], self.swing_leg_log[::100])):
            if leg != 'None':
                axes[3].scatter(t, phase, c=colors.get(leg, 'black'), s=10, alpha=0.5)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Phase (0-1)')
        axes[3].set_title('Gait Phase (FR=red, FL=blue, RR=green, RL=orange)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 图表已保存: {save_path}")
        
        plt.show()


def test_walk_gait():
    """测试Walk步态"""
    print("=" * 80)
    print("Walk步态测试（四拍步态）")
    print("=" * 80)
    
    # 创建机器人
    robot = LaikagoRobot(gui=True, start_height=0.5)
    
    # 创建Walk控制器
    walk_controller = WalkGaitController(robot)
    
    # 运行Walk步态
    results = walk_controller.run_walk(duration=15.0, verbose=True)
    
    # 绘制分析图
    walk_controller.plot_analysis(save_path='walk_gait_analysis.png')
    
    # 评估性能
    print("\n" + "=" * 80)
    print("性能评估")
    print("=" * 80)
    
    if results['distance_forward'] > 0.5:
        print(f"✅ 前进距离 > 0.5m: 通过 ({results['distance_forward']:.3f}m)")
    else:
        print(f"⚠️  前进距离 = {results['distance_forward']:.3f}m: 偏少")
    
    if results['distance_lateral'] < 0.3:
        print(f"✅ 横向偏移 < 0.3m: 通过")
    else:
        print(f"⚠️  横向偏移 = {results['distance_lateral']:.3f}m: 偏大")
    
    if results['height_std'] < 0.02:
        print(f"✅ 高度稳定 (std={results['height_std']:.4f}m): 通过")
    else:
        print(f"⚠️  高度波动 = {results['height_std']:.4f}m: 较大")
    
    if 0.18 < results['avg_height'] < 0.25:
        print(f"✅ 平均高度稳定: 通过")
    else:
        print(f"⚠️  平均高度 = {results['avg_height']:.3f}m: 异常")
    
    # 与Trot对比
    print("\n" + "=" * 80)
    print("与Trot步态对比")
    print("=" * 80)
    print(f"Walk速度: {results['forward_speed']:.3f} m/s")
    print(f"Trot速度: ~0.108 m/s (参考)")
    print(f"Walk稳定性预期更高（高度波动更小）")
    
    # 保持显示
    print("\n保持显示5秒...")
    for _ in range(5000):
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
    
    robot.close()
    
    return results


if __name__ == '__main__':
    results = test_walk_gait()
    
    print("\n" + "=" * 80)
    print("✅ Walk步态测试完成！")
    print("=" * 80)

