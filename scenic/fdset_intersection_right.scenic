"""路口右转 / intersection right turn. PDM route command: RIGHT.
Keywords: 右转, 路口右转, intersection, right turn, turning.
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
rightManeuvers = filter(lambda m: m.type == ManeuverType.RIGHT_TURN, startLane.maneuvers)
ego_maneuver = Uniform(*rightManeuvers)
param fdset_route_goal = ego_maneuver.endLane.centerline[-1]
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]
ego_approach = ego_maneuver.startLane.centerline[-1]
ego = Car following roadDirection from ego_approach for Range(-28, -20),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(ego_trajectory)
