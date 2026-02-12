#!/usr/bin/env python3
"""
改进的Trot步态控制器
目标：减少横向偏移(<0.3m)，提高前进速度
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


class ImprovedTrotController:
    """改进的Trot步态控制器，包含CoM控制"""
    
    def __init__(self, robot: LaikagoRobot):
        """
        初始化
        
        Args:
            robot: LaikagoRobot实例
        """
        self.robot = robot
        
        # Trot步态参数
        self.frequency = 1.0          # Hz
        self.stance_angle = 1.0       # 支撑相大腿角度
        self.swing_angle = 0.7        # 摆动相大腿角度
        self.knee_angle = -2.0        # 小腿固定角度
        
        # CoM控制参数
        self.target_com_y = 0.0       # 目标横向CoM位置
        self.com_kp = 1.0             # CoM比例增益（10倍增强）
        self.hip_abduction_max = 0.3  # 髋关节外展最大角度（增加范围）
        
        # 数据记录
        self.time_log = []
        self.com_log = []
        self.base_pos_log = []
        self.phase_log = []
        
    def get_com_position(self):
        """
        计算重心（CoM）位置
        简化版本：假设所有质量集中在基座
        """
        state = self.robot.get_state()
        return state['base_pos']
    
    def compute_hip_abduction_correction(self, current_com_y):
        """
        计算髋关节外展修正以控制横向CoM
        
        Args:
            current_com_y: 当前横向CoM位置
        
        Returns:
            correction: 髋关节外展修正角度 (4,)
        """
        # PD控制横向CoM
        error = self.target_com_y - current_com_y
        correction_angle = self.com_kp * error
        
        # 限制修正幅度
        correction_angle = np.clip(
            correction_angle,
            -self.hip_abduction_max,
            self.hip_abduction_max
        )
        
        # FR和FL向右修正，RR和RL向左修正（对称）
        # 实际上所有腿应该同向修正来移动CoM
        corrections = np.array([
            correction_angle,   # FR
            correction_angle,   # FL
            correction_angle,   # RR
            correction_angle    # RL
        ])
        
        return corrections
    
    def generate_trot_action(self, t):
        """
        生成Trot步态动作
        
        Args:
            t: 当前时间 (秒)
        
        Returns:
            action: 关节角度 (12,)
            phase: 当前步态相位 (0-1)
        """
        # 计算步态相位
        phase = (t * self.frequency) % 1.0
        
        # 对角步态: FR+RL一组, FL+RR一组
        if phase < 0.5:
            # FR+RL在支撑相, FL+RR在摆动相
            fr_rl_hip = self.stance_angle
            fl_rr_hip = self.swing_angle
        else:
            # FR+RL在摆动相, FL+RR在支撑相
            fr_rl_hip = self.swing_angle
            fl_rr_hip = self.stance_angle
        
        # 获取当前CoM位置
        com_pos = self.get_com_position()
        
        # 计算髋关节外展修正
        hip_corrections = self.compute_hip_abduction_correction(com_pos[1])
        
        # 构造目标角度（包含CoM修正）
        action = np.array([
            hip_corrections[0], fr_rl_hip, self.knee_angle,  # FR
            hip_corrections[1], fl_rr_hip, self.knee_angle,  # FL
            hip_corrections[2], fl_rr_hip, self.knee_angle,  # RR
            hip_corrections[3], fr_rl_hip, self.knee_angle   # RL
        ])
        
        # 记录数据
        self.time_log.append(t)
        self.com_log.append(com_pos.copy())
        self.base_pos_log.append(com_pos.copy())
        self.phase_log.append(phase)
        
        return action, phase
    
    def run_trot(self, duration=10.0, verbose=True):
        """
        运行Trot步态
        
        Args:
            duration: 持续时间 (秒)
            verbose: 是否打印信息
        
        Returns:
            results: 包含性能指标的字典
        """
        # 重置机器人
        self.robot.reset()
        
        # 清空日志
        self.time_log = []
        self.com_log = []
        self.base_pos_log = []
        self.phase_log = []
        
        if verbose:
            print(f"\n🏃 开始Trot步态 (持续{duration}秒)")
            print(f"   频率: {self.frequency} Hz")
            print(f"   CoM控制: Kp={self.com_kp}")
        
        t = 0
        dt = 0.001
        steps = int(duration / dt)
        
        for i in range(steps):
            # 生成动作
            action, phase = self.generate_trot_action(t)
            
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
                      f"X={pos_x:.3f}m, Y={pos_y:.3f}m, 相位={phase:.2f}")
        
        # 最终评估
        final_state = self.robot.get_state()
        height = final_state['base_pos'][2]
        distance_x = final_state['base_pos'][0]
        distance_y = abs(final_state['base_pos'][1])
        
        # 计算平均横向偏移
        com_array = np.array(self.com_log)
        avg_lateral_drift = np.mean(np.abs(com_array[:, 1]))
        max_lateral_drift = np.max(np.abs(com_array[:, 1]))
        
        results = {
            'final_height': height,
            'distance_forward': distance_x,
            'distance_lateral': distance_y,
            'avg_lateral_drift': avg_lateral_drift,
            'max_lateral_drift': max_lateral_drift,
            'forward_speed': distance_x / duration,
            'duration': duration
        }
        
        if verbose:
            print(f"\n📊 Trot步态结果:")
            print(f"   前进距离: {distance_x:.3f}m")
            print(f"   横向偏移: {distance_y:.3f}m")
            print(f"   平均横向漂移: {avg_lateral_drift:.3f}m")
            print(f"   最大横向漂移: {max_lateral_drift:.3f}m")
            print(f"   前进速度: {results['forward_speed']:.3f}m/s")
            print(f"   最终高度: {height:.3f}m")
        
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
        com_array = np.array(self.com_log)
        phase_array = np.array(self.phase_log)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 子图1: XY轨迹
        axes[0].plot(com_array[:, 0], com_array[:, 1], 'b-', linewidth=2, label='CoM轨迹')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='目标Y=0')
        axes[0].set_xlabel('X位置 (m)')
        axes[0].set_ylabel('Y位置 (m)')
        axes[0].set_title('重心(CoM)轨迹 - 俯视图')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # 子图2: 横向偏移随时间变化
        axes[1].plot(time_array, com_array[:, 1], 'b-', linewidth=2, label='横向偏移')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='目标')
        axes[1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='目标阈值±0.3m')
        axes[1].axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
        axes[1].fill_between(time_array, -0.3, 0.3, alpha=0.1, color='green')
        axes[1].set_xlabel('时间 (s)')
        axes[1].set_ylabel('横向偏移 (m)')
        axes[1].set_title('横向偏移随时间变化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 子图3: 步态相位
        axes[2].plot(time_array, phase_array, 'g-', linewidth=1, label='步态相位')
        axes[2].set_xlabel('时间 (s)')
        axes[2].set_ylabel('相位 (0-1)')
        axes[2].set_title('步态相位')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 图表已保存: {save_path}")
        
        plt.show()


def test_improved_trot():
    """测试改进的Trot步态"""
    print("=" * 80)
    print("改进的Trot步态测试")
    print("=" * 80)
    
    # 创建机器人
    robot = LaikagoRobot(gui=True, start_height=0.5)
    
    # 创建改进的Trot控制器
    trot_controller = ImprovedTrotController(robot)
    
    # 运行Trot步态
    results = trot_controller.run_trot(duration=10.0, verbose=True)
    
    # 绘制分析图
    trot_controller.plot_analysis(save_path='improved_trot_analysis.png')
    
    # 评估性能
    print("\n" + "=" * 80)
    print("性能评估")
    print("=" * 80)
    
    if results['distance_lateral'] < 0.3:
        print("✅ 横向偏移 < 0.3m: 通过")
    else:
        print(f"❌ 横向偏移 = {results['distance_lateral']:.3f}m: 未达标")
    
    if results['forward_speed'] > 0.15:
        print(f"✅ 前进速度 > 0.15m/s: 通过")
    else:
        print(f"⚠️  前进速度 = {results['forward_speed']:.3f}m/s: 偏慢")
    
    if 0.18 < results['final_height'] < 0.25:
        print("✅ 高度稳定: 通过")
    else:
        print(f"⚠️  高度 = {results['final_height']:.3f}m: 异常")
    
    # 保持显示
    print("\n保持显示5秒...")
    for _ in range(5000):
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
    
    robot.close()
    
    return results


if __name__ == '__main__':
    results = test_improved_trot()
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)

