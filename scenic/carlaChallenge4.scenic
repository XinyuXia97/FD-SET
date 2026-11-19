"""路口自行车横穿 / cyclist crossing the exit lane of a sampled intersection maneuver.
Keywords: 自行车, 路口, 横穿, bicycle, cyclist, intersection crossing.
"""

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('../maps/Town_Safebench_Light.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"
BICYCLE_MIN_SPEED = 1.5
THRESHOLD = 18
BRAKE_ACTION = 1.0
SAFETY_DISTANCE = 10

## DEFINING BEHAVIORS
behavior EgoBehavior(trajectory):
    try:
        do FollowTrajectoryBehavior(trajectory = trajectory)

    interrupt when withinDistanceToObjsInLane(self, SAFETY_DISTANCE):
        take SetBrakeAction(BRAKE_ACTION)

# CARLA bicycle blueprints ignore throttle and velocity commands. Move the real
# actor by a fixed distance each simulation tick after the existing trigger.
behavior BicycleBehavior(speed=3, threshold=15):
    while (distance from self to ego) > threshold:
        wait
    while True:
        take OffsetAction(0 @ (speed * 0.1))

## DEFINING SPATIAL RELATIONS
# make sure to put '*' to uniformly randomly select from all elements of the list
intersec = Uniform(*network.intersections)
startLane = Uniform(*intersec.incomingLanes)
maneuver = Uniform(*startLane.maneuvers)
param fdset_route_goal = maneuver.endLane.centerline[-1]
ego_trajectory = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]

spot = OrientedPoint at maneuver.startLane.centerline[-1],
    facing roadDirection
ego = Car following roadDirection from spot for Range(-24, -18),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(trajectory = ego_trajectory)

spotBicycle = OrientedPoint at maneuver.endLane.centerline[0],
    facing roadDirection
bicycle = Bicycle at spotBicycle offset by 3.5@0,
    with heading 90 deg relative to spotBicycle.heading,
    with rolename "fdset_kinematic_bicycle",
    with behavior BicycleBehavior(BICYCLE_MIN_SPEED, THRESHOLD),
    with regionContainedIn None
