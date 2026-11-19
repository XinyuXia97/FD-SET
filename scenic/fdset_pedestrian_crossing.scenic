"""行人横穿 / pedestrian crossing at a mid-block road segment.
Keywords: 行人横穿, 行人, pedestrian, walker, crossing, vulnerable road user.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 7
WALK_SPEED = 1.5
CROSSING_THRESHOLD = 18

behavior EgoBehavior():
    do FollowLaneBehavior(EGO_SPEED)

behavior PedestrianBehavior():
    while (distance from self to ego) > CROSSING_THRESHOLD:
        wait
    while True:
        do WalkForwardBehavior(WALK_SPEED)

lane = Uniform(*network.lanes)
spawn = OrientedPoint on lane.centerline,
    facing roadDirection
ego = Car following roadDirection from spawn for Range(-26, -20),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()
require (distance to intersection) > 60
pedestrian = Pedestrian at spawn offset by 3.5@0,
    with heading 90 deg relative to spawn.heading,
    with behavior PedestrianBehavior(),
    with regionContainedIn None
require (pedestrian.position - ego.position).rotatedBy(-ego.heading).y > 12
