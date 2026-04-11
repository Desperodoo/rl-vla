cmake_minimum_required(VERSION 3.0.2)
project(carm_deploy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_package()

# 安装 Python 脚本 (主程序)
catkin_install_python(PROGRAMS
  inference/inference_ros.py
  data/record_data_ros.py
  data/teleop_shadow_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装 launch 文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# 安装 Python 模块
install(DIRECTORY 
  core/
  inference/
  data/
  tools/
  utils/
  camera/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
)
