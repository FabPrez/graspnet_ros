cmake_minimum_required(VERSION 3.8)
project(graspnet_ros)

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)



# In order to add the executable that can be used by ros2
# add_executable(nbv_pipeline_node
#   src/nbv_pipeline.cpp
#   src/nbv_pipeline_node.cpp
# )
# ament_target_dependencies(nbv_pipeline_node
#   rclcpp rclcpp_components
#   std_msgs std_srvs
#   sensor_msgs
#   nbv_octomap_utils octomap_msgs 
#   pcl_conversions pcl_ros
#   pointcloud_handler
# )

# # In order to add the library that can be used by ros2
# set(library_name octomap_utils_library)
# add_library(${library_name} SHARED
#   src/octomap_utils.cpp
#   src/octomap_utils_node.cpp
# )
# ament_target_dependencies(${library_name}
#   rclcpp rclcpp_components
#   std_msgs std_srvs
#   sensor_msgs
#   octomap octomap_msgs octomap_server
#   pcl_conversions pcl_ros
#   pointcloud_handler
# )

# In order to let ros2 find the executable
install(PROGRAMS
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/graspnet-baseline
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/main.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/venv_graspnet
  DESTINATION share/${PROJECT_NAME}
)


# Installa i launch file
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# # In order to let ros2 find the library
# install(TARGETS ${library_name}
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

# In order to let ros2 find the folders
# install( 
#   DIRECTORY include launch src
#   DESTINATION share/${PROJECT_NAME}
# )


# In order to link some needed libraries into this package
# target_link_libraries(nbv_pipeline_node
#   ${PCL_LIBRARIES}
#   ${OCTOMAP_LIBRARIES}
# )

# target_link_libraries(${library_name}
#   ${PCL_LIBRARIES}
#   ${OCTOMAP_LIBRARIES}
# )


# # In order to let the library be visible from other packages
# install(
#   DIRECTORY include/
#   DESTINATION include
# )

# ament_export_include_directories(include)
# ament_export_libraries(${library_name})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()