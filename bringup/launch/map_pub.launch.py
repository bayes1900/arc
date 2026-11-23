from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pcdPath = LaunchConfiguration('pcd_path')
    addGround = LaunchConfiguration('add_ground')
    droneId = LaunchConfiguration('drone_id')
    initX = LaunchConfiguration('init_x')
    initY = LaunchConfiguration('init_y')
    initZ = LaunchConfiguration('init_z')
    publishRate = LaunchConfiguration('publish_rate')
    useRviz = LaunchConfiguration('rviz')

    defaultPcdPath = PathJoinSubstitution([
        FindPackageShare('arc_sim'), 'resource', 'pillar.pcd'
    ])
    rvizConfig = PathJoinSubstitution([
        FindPackageShare('arc_bringup'), 'config', 'rviz', 'map_pub.rviz'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'pcd_path',
            default_value=defaultPcdPath,
            description='PCD file published as the global map'
        ),
        DeclareLaunchArgument(
            'add_ground',
            default_value='true',
            description='Append a generated ground plane to the map'
        ),
        DeclareLaunchArgument(
            'drone_id',
            default_value='1',
            description='Drone identifier used by odom_pub'
        ),
        DeclareLaunchArgument(
            'init_x',
            default_value='0.0',
            description='Initial x position of the drone'
        ),
        DeclareLaunchArgument(
            'init_y',
            default_value='0.0',
            description='Initial y position of the drone'
        ),
        DeclareLaunchArgument(
            'init_z',
            default_value='1.0',
            description='Initial z position of the drone'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='100.0',
            description='Odometry publish rate in Hz'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz2 with the map visualization'
        ),
        Node(
            package='arc_sim',
            executable='map_pub',
            name='map_pub',
            output='screen',
            parameters=[{
                'pcdPath': pcdPath,
                'addGround': ParameterValue(addGround, value_type=bool),
            }],
        ),
        Node(
            package='arc_sim',
            executable='odom_pub',
            name='odom_pub',
            output='screen',
            parameters=[{
                'drone_id': ParameterValue(droneId, value_type=int),
                'init_x': ParameterValue(initX, value_type=float),
                'init_y': ParameterValue(initY, value_type=float),
                'init_z': ParameterValue(initZ, value_type=float),
                'publish_rate': ParameterValue(publishRate, value_type=float),
            }],
        ),
        Node(
            package='arc_sim',
            executable='dronevisual_pub',
            name='dronevisual_pub',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rvizConfig],
            condition=IfCondition(useRviz),
        ),
    ])
