/**
 * 三次样条插值器测试
 */

#include "cubic_interpolator.hpp"
#include "trajectory_buffer.hpp"
#include "../shared/shm_protocol.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

void test_cubic_spline_1d() {
    std::cout << "\n=== 测试 CubicSpline1D ===" << std::endl;
    
    rtc::CubicSpline1D spline;
    
    // 简单的二次函数: y = x^2, x in [0, 1, 2, 3]
    double t[] = {0.0, 1.0, 2.0, 3.0};
    double y[] = {0.0, 1.0, 4.0, 9.0};
    
    spline.build(t, y, 4, true);
    
    std::cout << "  原始点: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << "(" << t[i] << ", " << y[i] << ") ";
    }
    std::cout << std::endl;
    
    std::cout << "  插值结果:" << std::endl;
    for (double x = 0.0; x <= 3.0; x += 0.5) {
        double y_interp = spline.eval(x);
        double y_true = x * x;
        std::cout << "    t=" << std::fixed << std::setprecision(1) << x 
                  << ": interp=" << std::setprecision(4) << y_interp
                  << ", true=" << y_true 
                  << ", error=" << std::abs(y_interp - y_true) << std::endl;
    }
}

void test_cubic_interpolator() {
    std::cout << "\n=== 测试 CubicInterpolator ===" << std::endl;
    
    rtc::CubicInterpolator interp;
    
    // 8 个关键帧，3 个关节
    constexpr int H = 8;
    constexpr int dof = 3;
    double dt_key = 1.0 / 30.0;  // 30Hz
    double t0 = 0.0;
    
    // 构造关键帧: 简单的正弦运动
    double q_key[H][rtc::CubicInterpolator::MAX_DOF];
    for (int i = 0; i < H; ++i) {
        double t = i * dt_key;
        q_key[i][0] = std::sin(2 * M_PI * t);  // 1Hz 正弦
        q_key[i][1] = std::cos(2 * M_PI * t);
        q_key[i][2] = t;  // 线性
    }
    
    interp.build(q_key, H, dof, dt_key, t0);
    
    std::cout << "  H=" << H << ", dof=" << dof << ", dt_key=" << dt_key << "s" << std::endl;
    std::cout << "  轨迹时长: " << interp.t_end() - interp.t0() << "s" << std::endl;
    
    // 500Hz 采样
    double sample_dt = 1.0 / 500.0;
    std::cout << "\n  500Hz 采样 (前 10 个点):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        double t = t0 + i * sample_dt;
        double q[3];
        interp.eval(t, q);
        std::cout << "    t=" << std::fixed << std::setprecision(4) << t 
                  << ": q=[" << std::setprecision(4) << q[0] 
                  << ", " << q[1] << ", " << q[2] << "]" << std::endl;
    }
}

void test_trajectory_buffer() {
    std::cout << "\n=== 测试 TrajectoryBuffer ===" << std::endl;
    
    rtc::TrajectoryBufferConfig config;
    config.commit_time = 0.10;
    config.blend_window = 0.03;
    config.dof = 3;
    
    rtc::TrajectoryBuffer buffer(config);
    
    // 构造关键帧
    constexpr int H = 8;
    double dt_key = 1.0 / 30.0;
    double t_now = rtc::get_monotonic_time();
    
    double q_key[H][rtc::CubicInterpolator::MAX_DOF];
    for (int i = 0; i < H; ++i) {
        q_key[i][0] = 0.1 * i;
        q_key[i][1] = 0.2 * i;
        q_key[i][2] = 0.3 * i;
    }
    
    // 更新轨迹
    buffer.update_simple(q_key, H, dt_key, t_now);
    
    std::cout << "  轨迹已更新, t_now=" << t_now << std::endl;
    std::cout << "  has_trajectory=" << buffer.has_trajectory() << std::endl;
    std::cout << "  remaining_time=" << buffer.remaining_time(t_now) << "s" << std::endl;
    
    // 采样
    double q[3];
    buffer.sample(t_now, q);
    std::cout << "  sample(t_now): q=[" << q[0] << ", " << q[1] << ", " << q[2] << "]" << std::endl;
    
    buffer.sample(t_now + 0.1, q);
    std::cout << "  sample(t_now+0.1): q=[" << q[0] << ", " << q[1] << ", " << q[2] << "]" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RTC 插值器测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_cubic_spline_1d();
    test_cubic_interpolator();
    test_trajectory_buffer();
    
    std::cout << "\n✓ 所有测试完成" << std::endl;
    
    return 0;
}
