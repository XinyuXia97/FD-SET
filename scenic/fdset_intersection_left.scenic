"""路口左转 / intersection left turn. PDM route command: LEFT; 典型无保护左转。
Keywords: 左转, 路口左转, intersection, left turn, turning.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 8

behavior EgoBehavior(trajectory):
    do FollowTrajectoryBehavior(target_speed=EGO_SPEED, trajectory=trajectory)

fourWayIntersections = filter(lambda i: i.is4Way, network.intersections)
intersection = Uniform(*fourWayIntersections)
startLane = Uniform(*intersection.incomingLanes)
leftManeuvers = filter(lambda m: m.type == ManeuverType.LEFT_TURN, startLane.maneuvers)
ego_maneuver = Uniform(*leftManeuvers)
param fdset_route_goal = ego_maneuver.endLane.centerline[-1]
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]
ego_approach = ego_maneuver.startLane.centerline[-1]
ego = Car following roadDirection from ego_approach for Range(-28, -20),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(ego_trajectory)
