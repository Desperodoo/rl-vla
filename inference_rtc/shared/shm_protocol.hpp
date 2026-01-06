#ifndef SHM_PROTOCOL_HPP
#define SHM_PROTOCOL_HPP

/**
 * 共享内存协议定义 (C++ 侧)
 * 
 * Python → C++ 单向无锁通信
 * 
 * 同步方式:
 *   Python: 写 payload → version++
 *   C++: 读 version1 → payload → version2, version1 == version2 才采用
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <array>
#include <optional>

namespace rtc {

/**
 * 共享内存布局常量
 */
struct ShmProtocol {
    static constexpr int VERSION = 1;
    
    // 默认参数
    static constexpr int DEFAULT_DOF = 7;       // 6 关节 + 1 夹爪
    static constexpr int DEFAULT_H = 8;         // 8 个关键帧
    static constexpr double DEFAULT_DT_KEY = 1.0 / 30.0;  // 30Hz
    
    // Header 布局 (32 bytes)
    static constexpr size_t OFFSET_VERSION = 0;   // uint64 (8 bytes)
    static constexpr size_t OFFSET_T_WRITE = 8;   // double (8 bytes)
    static constexpr size_t OFFSET_DOF = 16;      // int32 (4 bytes)
    static constexpr size_t OFFSET_H = 20;        // int32 (4 bytes)
    static constexpr size_t OFFSET_DT_KEY = 24;   // double (8 bytes)
    static constexpr size_t HEADER_SIZE = 32;
    
    // Payload 布局
    static constexpr size_t OFFSET_Q_KEY = 32;    // double[H][dof]
    
    static constexpr size_t calc_total_size(int H = DEFAULT_H, int dof = DEFAULT_DOF) {
        return HEADER_SIZE + H * dof * sizeof(double);
    }
};

/**
 * 关键帧数据
 */
struct KeyframeData {
    uint64_t version = 0;
    double t_write_mono = 0.0;
    int dof = ShmProtocol::DEFAULT_DOF;
    int H = ShmProtocol::DEFAULT_H;
    double dt_key = ShmProtocol::DEFAULT_DT_KEY;
    
    // q_key[H][dof] - 使用固定大小数组，避免动态分配
    // 最大支持 16 个关键帧，10 个自由度
    static constexpr int MAX_H = 16;
    static constexpr int MAX_DOF = 10;
    double q_key[MAX_H][MAX_DOF] = {{0.0}};
    
    bool valid() const { return version > 0; }
};

/**
 * 共享内存关键帧读取器 (C++ 伺服进程使用)
 */
class ShmKeyframeReader {
public:
    static constexpr const char* DEFAULT_SHM_NAME = "rtc_keyframes";
    
    ShmKeyframeReader() = default;
    
    /**
     * 连接到共享内存
     * @param name 共享内存名称
     * @param timeout_sec 超时时间 (秒)
     * @return 是否成功
     */
    bool connect(const std::string& name = DEFAULT_SHM_NAME, double timeout_sec = 10.0) {
        auto start = std::chrono::steady_clock::now();
        
        while (true) {
            // 尝试打开共享内存 (只读)
            std::string shm_path = "/dev/shm/" + name;
            _fd = shm_open(name.c_str(), O_RDONLY, 0666);
            
            if (_fd >= 0) {
                // 获取大小
                struct stat sb;
                if (fstat(_fd, &sb) == 0) {
                    _size = sb.st_size;
                    
                    // 映射到内存
                    _ptr = mmap(nullptr, _size, PROT_READ, MAP_SHARED, _fd, 0);
                    if (_ptr != MAP_FAILED) {
                        _connected = true;
                        std::cout << "[ShmReader] 连接成功: " << name 
                                  << ", size=" << _size << " bytes" << std::endl;
                        return true;
                    }
                }
                close(_fd);
                _fd = -1;
            }
            
            // 检查超时
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_sec) {
                std::cerr << "[ShmReader] 连接超时: " << name << std::endl;
                return false;
            }
            
            // 等待后重试
            usleep(100000);  // 100ms
        }
    }
    
    /**
     * 读取关键帧 (无锁)
     * @param data 输出数据
     * @return 是否读取到新数据 (version 变化)
     */
    bool read(KeyframeData& data) {
        if (!_connected || _ptr == nullptr) {
            return false;
        }
        
        const uint8_t* buf = static_cast<const uint8_t*>(_ptr);
        
        // 1. 读取 version1
        uint64_t version1;
        std::memcpy(&version1, buf + ShmProtocol::OFFSET_VERSION, sizeof(version1));
        
        // 如果版本没变，不需要更新
        if (version1 == _last_version) {
            return false;
        }
        
        // 如果版本为 0，说明还没有有效数据
        if (version1 == 0) {
            return false;
        }
        
        // 2. 读取 header
        std::memcpy(&data.t_write_mono, buf + ShmProtocol::OFFSET_T_WRITE, sizeof(double));
        std::memcpy(&data.dof, buf + ShmProtocol::OFFSET_DOF, sizeof(int32_t));
        std::memcpy(&data.H, buf + ShmProtocol::OFFSET_H, sizeof(int32_t));
        std::memcpy(&data.dt_key, buf + ShmProtocol::OFFSET_DT_KEY, sizeof(double));
        
        // 检查参数合法性
        if (data.H > KeyframeData::MAX_H || data.dof > KeyframeData::MAX_DOF) {
            std::cerr << "[ShmReader] 参数超限: H=" << data.H << ", dof=" << data.dof << std::endl;
            return false;
        }
        
        // 3. 读取 payload (q_key)
        size_t payload_size = data.H * data.dof * sizeof(double);
        // 按行拷贝，因为 Python 端是 C-order (row-major)
        const double* src = reinterpret_cast<const double*>(buf + ShmProtocol::OFFSET_Q_KEY);
        for (int i = 0; i < data.H; ++i) {
            for (int j = 0; j < data.dof; ++j) {
                data.q_key[i][j] = src[i * data.dof + j];
            }
        }
        
        // 4. 读取 version2
        uint64_t version2;
        std::memcpy(&version2, buf + ShmProtocol::OFFSET_VERSION, sizeof(version2));
        
        // 5. 检查一致性
        if (version1 != version2) {
            // 写入过程中被读取，数据不一致，丢弃
            return false;
        }
        
        // 成功读取
        data.version = version1;
        _last_version = version1;
        return true;
    }
    
    /**
     * 获取当前 version (不读取数据)
     */
    uint64_t peek_version() const {
        if (!_connected || _ptr == nullptr) {
            return 0;
        }
        const uint8_t* buf = static_cast<const uint8_t*>(_ptr);
        uint64_t version;
        std::memcpy(&version, buf + ShmProtocol::OFFSET_VERSION, sizeof(version));
        return version;
    }
    
    /**
     * 获取上次读取的 t_write_mono
     */
    double get_last_t_write() const {
        if (!_connected || _ptr == nullptr) {
            return 0.0;
        }
        const uint8_t* buf = static_cast<const uint8_t*>(_ptr);
        double t_write;
        std::memcpy(&t_write, buf + ShmProtocol::OFFSET_T_WRITE, sizeof(double));
        return t_write;
    }
    
    bool connected() const { return _connected; }
    
    void disconnect() {
        if (_ptr != nullptr && _ptr != MAP_FAILED) {
            munmap(_ptr, _size);
            _ptr = nullptr;
        }
        if (_fd >= 0) {
            close(_fd);
            _fd = -1;
        }
        _connected = false;
    }
    
    ~ShmKeyframeReader() {
        disconnect();
    }
    
private:
    int _fd = -1;
    void* _ptr = nullptr;
    size_t _size = 0;
    bool _connected = false;
    uint64_t _last_version = 0;
};

/**
 * 获取单调时钟时间 (CLOCK_MONOTONIC)
 */
inline double get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

}  // namespace rtc

#endif  // SHM_PROTOCOL_HPP
