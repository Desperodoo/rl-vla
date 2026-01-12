cmake_minimum_required(VERSION 3.0.2)
project(carm_ros_deploy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_package()

# 安装 Python 脚本
catkin_install_python(PROGRAMS
  inference_ros.py
  record_data_ros.py
  env_ros.py
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

# 安装 utils 模块
install(DIRECTORY utils/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/utils
)
