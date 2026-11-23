import copy
import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


defaultPositions = [
    (0.0, 0.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
    (0.0, -2.0),
]


def loadSection(configPath, section):
    with open(configPath, 'r') as stream:
        data = yaml.safe_load(stream)
    return copy.deepcopy(data[section]['ros__parameters'])


def toBool(value):
    return str(value).lower() in ('true', '1', 'yes', 'on')


def launchNodes(context):
    droneNum = max(1, int(LaunchConfiguration('drone_num').perform(context)))
    autoTrigger = toBool(LaunchConfiguration('auto_trigger').perform(context))
    finishUnreachableFrontier = toBool(
        LaunchConfiguration('finish_unreachable_frontier').perform(context)
    )
    useRviz = toBool(LaunchConfiguration('rviz').perform(context))
    addGround = toBool(LaunchConfiguration('add_ground').perform(context))
    useVio = toBool(LaunchConfiguration('use_vio').perform(context))
    useVioForPlanning = toBool(
        LaunchConfiguration('use_vio_for_planning').perform(context)
    )
    useLio = toBool(LaunchConfiguration('use_lio').perform(context))
    useLioForPlanning = toBool(
        LaunchConfiguration('use_lio_for_planning').perform(context)
    )
    useEstimation = useVio or useLio
    pcdPath = LaunchConfiguration('pcd_path').perform(context)
    vioConfig = LaunchConfiguration('vio_config').perform(context)
    vioCalib = LaunchConfiguration('vio_calib').perform(context)
    lioConfig = LaunchConfiguration('lio_config').perform(context)

    bringupShare = get_package_share_directory('arc_bringup')
    configPath = LaunchConfiguration('config').perform(context)
    rvizConfig = os.path.join(bringupShare, 'config', 'rviz', 'swarm_exploration.rviz')

    explorationBase = loadSection(configPath, 'exploration_node')
    renderBase = loadSection(configPath, 'pcl_render_node')
    odomBase = loadSection(configPath, 'odom_pub')
    visualBase = loadSection(configPath, 'odom_visualization')
    lidarBase = loadSection(configPath, 'lidar_render_node') if useLio else {}

    actions = [
        Node(
            package='arc_sim',
            executable='map_pub',
            name='map_pub',
            output='screen',
            parameters=[{
                'pcdPath': pcdPath,
                'addGround': addGround or useLio,
            }],
        )
    ]

    for droneId in range(1, droneNum + 1):
        initX, initY = defaultPositions[(droneId - 1) % len(defaultPositions)]

        gtOdom = '/odom_gt_{0}'.format(droneId) if useEstimation else '/odom_world_{0}'.format(droneId)
        planningOdom = (
            '/lio/odom_{0}'.format(droneId) if useLioForPlanning else
            ('/vio/odom_{0}'.format(droneId) if useVioForPlanning else gtOdom)
        )

        explorationParams = copy.deepcopy(explorationBase)
        explorationParams['exploration']['droneId'] = droneId
        explorationParams['exploration']['droneNum'] = droneNum
        explorationParams['exploration']['autoTrigger'] = autoTrigger
        explorationParams['exploration']['finishUnreachableFrontier'] = finishUnreachableFrontier
        tspWorkDir = '/tmp/arc_exploration_{0}'.format(droneId)
        explorationParams['tsp']['workDir'] = tspWorkDir

        actions.append(Node(
            package='arc_sim',
            executable='lkh_solver',
            name='tsp_solver_{0}'.format(droneId),
            output='log',
            parameters=[{
                'drone_id': droneId,
                'problem_id': 1,
                'work_dir': tspWorkDir,
            }],
        ))
        actions.append(Node(
            package='arc_sim',
            executable='lkh_solver',
            name='acvrp_solver_{0}'.format(droneId),
            output='log',
            parameters=[{
                'drone_id': droneId,
                'problem_id': 2,
                'work_dir': tspWorkDir,
            }],
        ))

        odomParams = copy.deepcopy(odomBase)
        odomParams['drone_id'] = droneId
        odomParams['init_x'] = initX
        odomParams['init_y'] = initY
        odomParams['init_z'] = 1.0

        visualParams = copy.deepcopy(visualBase)
        visualParams['drone_id'] = droneId
        # RACER colors four active drones against a five-drone palette.
        visualParams['drone_num'] = max(droneNum, 5)

        actions.append(Node(
            package='arc_sim',
            executable='odom_pub',
            name='odom_pub_{0}'.format(droneId),
            output='screen',
            parameters=[odomParams],
            remappings=[
                ('command', '/planning/pos_cmd_{0}'.format(droneId)),
                ('odometry', gtOdom),
                ('imu', '/sensor/imu_{0}'.format(droneId)),
            ],
        ))
        actions.append(Node(
            package='arc_sim',
            executable='pcl_render_node',
            name='pcl_render_node_{0}'.format(droneId),
            output='screen',
            parameters=[copy.deepcopy(renderBase)],
            remappings=[
                ('global_map', '/map_generator/global_cloud'),
                ('odometry', gtOdom),
                ('depth', '/sensor/depth_{0}'.format(droneId)),
                ('image', '/sensor/image_{0}'.format(droneId)),
                ('sensor_pose', '/sensor/pose_{0}'.format(droneId)),
            ],
        ))
        if useLio:
            actions.append(Node(
                package='arc_sim',
                executable='lidar_render_node',
                name='lidar_render_node_{0}'.format(droneId),
                output='screen',
                parameters=[copy.deepcopy(lidarBase), {
                    'drone_id': droneId,
                    'lidar_frame': 'lidar_{0}'.format(droneId),
                }],
                remappings=[
                    ('global_map', '/map_generator/global_cloud'),
                    ('odometry', gtOdom),
                    ('scan', '/sensor/lidar_{0}'.format(droneId)),
                ],
            ))
            actions.append(Node(
                package='arc_sim',
                executable='lio_node',
                name='lio_node_{0}'.format(droneId),
                output='screen',
                parameters=[{
                    'config_file': lioConfig,
                    'lidar_topic': '/sensor/lidar_{0}'.format(droneId),
                    'imu_topic': '/sensor/imu_{0}'.format(droneId),
                    'world_frame': 'world',
                    'body_frame': 'imu_{0}'.format(droneId),
                    'publish_tf': True,
                    'publish_cloud': True,
                    'initial_pose': [initX, initY, 1.0, 0.0],
                }],
                remappings=[
                    ('odom', '/lio/odom_{0}'.format(droneId)),
                    ('cloud_registered', '/lio/cloud_{0}'.format(droneId)),
                ],
            ))
        if useVio:
            actions.append(Node(
                package='arc_sim',
                executable='vio_node',
                name='vio_node_{0}'.format(droneId),
                output='screen',
                parameters=[dict({
                    'config_file': vioConfig,
                    'calib_file': vioCalib,
                    'image_topics': ['/sensor/image_{0}'.format(droneId)],
                    'imu_topic': '/sensor/imu_{0}'.format(droneId),
                    'world_frame': 'world',
                    'body_frame': 'imu_{0}'.format(droneId),
                    'publish_tf': True,
                }, **({
                    'initial_pose': [initX, initY, 1.0, 0.0, 0.0, 0.0],
                } if useVioForPlanning else {}))],
                remappings=[
                    ('odom', '/vio/odom_{0}'.format(droneId)),
                ],
            ))
        actions.append(Node(
            package='arc_sim',
            executable='exploration_node',
            name='exploration_node_{0}'.format(droneId),
            output='log',
            parameters=[explorationParams],
            remappings=[
                ('depth', '/sensor/depth_{0}'.format(droneId)),
                ('pose', '/sensor/pose_{0}'.format(droneId)),
                ('odometry', planningOdom),
                ('/exploration/state', '/exploration/state_{0}'.format(droneId)),
                ('/exploration/events', '/exploration/events_{0}'.format(droneId)),
                ('/exploration/metrics', '/exploration/metrics_{0}'.format(droneId)),
                ('/exploration/occupancy_local', '/sdf_map/occupancy_local_{0}'.format(droneId)),
                ('/exploration/occupancy_all', '/sdf_map/occupancy_all_{0}'.format(droneId)),
                ('/exploration/unknown', '/sdf_map/unknown_{0}'.format(droneId)),
                ('/exploration/occupancy_inflate', '/sdf_map/occupancy_local_inflate_{0}'.format(droneId)),
                ('/exploration/depth_cloud', '/sdf_map/depth_cloud_{0}'.format(droneId)),
                ('/planning_vis/frontier', '/planning_vis/frontier_{0}'.format(droneId)),
                ('/planning_vis/viewpoints', '/planning_vis/viewpoints_{0}'.format(droneId)),
                ('/planning_vis/trajectory', '/planning_vis/trajectory_{0}'.format(droneId)),
                ('/planning/bspline', '/planning/bspline_{0}'.format(droneId)),
            ],
        ))
        actions.append(Node(
            package='arc_sim',
            executable='traj_server',
            name='traj_server_{0}'.format(droneId),
            output='log',
            parameters=[{
                'drone_id': droneId,
                'drone_num': max(droneNum, 5),
                'diagnostics': explorationBase.get('diagnostics', {'enabled': True}),
            }],
            remappings=[
                ('trajectory', '/planning/bspline_{0}'.format(droneId)),
                ('position_cmd', '/planning/pos_cmd_{0}'.format(droneId)),
                ('position_cmd_vis', '/planning/position_cmd_vis_{0}'.format(droneId)),
                ('travel_traj', '/planning/travel_traj_{0}'.format(droneId)),
                ('odometry', planningOdom),
                ('/traj_server/events', '/traj_server/events_{0}'.format(droneId)),
                ('/traj_server/metrics', '/traj_server/metrics_{0}'.format(droneId)),
            ],
        ))
        actions.append(Node(
            package='arc_sim',
            executable='dronevisual_pub',
            name='odom_visualization_{0}'.format(droneId),
            output='screen',
            parameters=[visualParams],
            remappings=[
                ('odometry', gtOdom),
            ],
        ))

    if useRviz:
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rvizConfig],
        ))
    return actions


def generate_launch_description():
    bringupShare = get_package_share_directory('arc_bringup')
    defaultPcdPath = os.path.join(get_package_share_directory('arc_sim'), 'resource', 'pillar.pcd')
    defaultConfig = os.path.join(bringupShare, 'config', 'exploration.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'drone_num',
            default_value='4',
            description='Number of drones in the swarm',
        ),
        DeclareLaunchArgument(
            'pcd_path',
            default_value=defaultPcdPath,
            description='PCD file published as the global map',
        ),
        DeclareLaunchArgument(
            'add_ground',
            default_value='false',
            description='Append a generated ground plane to the map',
        ),
        DeclareLaunchArgument(
            'config',
            default_value=defaultConfig,
            description='Exploration node parameter file',
        ),
        DeclareLaunchArgument(
            'auto_trigger',
            default_value='true',
            description='Start exploration automatically after initialization',
        ),
        DeclareLaunchArgument(
            'finish_unreachable_frontier',
            default_value='false',
            description='Enter IDLE when the last reachable frontier cannot be covered',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz2 with the swarm exploration visualization',
        ),
        DeclareLaunchArgument(
            'use_vio',
            default_value='false',
            description='Run the VIO node and publish odometry on /vio/odom_{id}',
        ),
        DeclareLaunchArgument(
            'use_vio_for_planning',
            default_value='false',
            description='Use VIO odometry for exploration instead of ground truth',
        ),
        DeclareLaunchArgument(
            'vio_config',
            default_value=os.path.join(bringupShare, 'config', 'vio_config.yaml'),
            description='VIO configuration file',
        ),
        DeclareLaunchArgument(
            'vio_calib',
            default_value=os.path.join(bringupShare, 'config', 'vio_calib.yaml'),
            description='VIO calibration file',
        ),
        DeclareLaunchArgument(
            'use_lio',
            default_value='false',
            description='Run the LiDAR-inertial odometry node and publish on /lio/odom_{id}',
        ),
        DeclareLaunchArgument(
            'use_lio_for_planning',
            default_value='false',
            description='Use LIO odometry for exploration instead of ground truth',
        ),
        DeclareLaunchArgument(
            'lio_config',
            default_value=os.path.join(bringupShare, 'config', 'lio.yaml'),
            description='LIO configuration file',
        ),
        OpaqueFunction(function=launchNodes),
    ])
