"""路口直行 / intersection straight-through. PDM route: LANEFOLLOW on a straight connector.
Keywords: 直行, 路口直行, intersection, straight, straight through.
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
straightManeuvers = filter(lambda m: m.type == ManeuverType.STRAIGHT, startLane.maneuvers)
ego_maneuver = Uniform(*straightManeuvers)
param fdset_route_goal = ego_maneuver.endLane.centerline[-1]
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]
ego_approach = ego_maneuver.startLane.centerline[-1]
ego = Car following roadDirection from ego_approach for Range(-28, -20),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(ego_trajectory)
