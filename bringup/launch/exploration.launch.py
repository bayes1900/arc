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
    configFile = LaunchConfiguration('config')
    useRviz = LaunchConfiguration('rviz')
    initX = LaunchConfiguration('init_x')
    initY = LaunchConfiguration('init_y')
    initZ = LaunchConfiguration('init_z')

    defaultPcdPath = PathJoinSubstitution([
        FindPackageShare('arc_sim'), 'resource', 'pillar.pcd'
    ])
    defaultConfig = PathJoinSubstitution([
        FindPackageShare('arc_bringup'), 'config', 'exploration.yaml'
    ])
    rvizConfig = PathJoinSubstitution([
        FindPackageShare('arc_bringup'), 'config', 'rviz', 'exploration.rviz'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'pcd_path',
            default_value=defaultPcdPath,
            description='PCD file published as the global map'
        ),
        DeclareLaunchArgument(
            'add_ground',
            default_value='false',
            description='Append a generated ground plane to the map'
        ),
        DeclareLaunchArgument(
            'config',
            default_value=defaultConfig,
            description='Exploration node parameter file'
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
            'rviz',
            default_value='true',
            description='Start RViz2 with the exploration visualization'
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
            parameters=[configFile, {
                'init_x': ParameterValue(initX, value_type=float),
                'init_y': ParameterValue(initY, value_type=float),
                'init_z': ParameterValue(initZ, value_type=float),
            }],
            remappings=[
                ('command', '/planning/pos_cmd_1'),
                ('odometry', '/odom_world_1'),
            ],
        ),
        Node(
            package='arc_sim',
            executable='pcl_render_node',
            name='pcl_render_node',
            output='screen',
            parameters=[configFile],
            remappings=[
                ('global_map', '/map_generator/global_cloud'),
                ('odometry', '/odom_world_1'),
                ('depth', '/sensor/depth'),
                ('sensor_pose', '/sensor/pose'),
            ],
        ),
        Node(
            package='arc_sim',
            executable='exploration_node',
            name='exploration_node',
            output='log',
            parameters=[configFile],
            remappings=[
                ('depth', '/sensor/depth'),
                ('pose', '/sensor/pose'),
                ('odometry', '/odom_world_1'),
                ('command', '/planning/pos_cmd_1'),
            ],
        ),
        Node(
            package='arc_sim',
            executable='dronevisual_pub',
            name='dronevisual_pub',
            output='screen',
            remappings=[
                ('odometry', '/odom_world_1'),
            ],
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
