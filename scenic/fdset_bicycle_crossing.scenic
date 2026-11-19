"""非机动车横穿 / cyclist crossing an urban road mid-block.
Keywords: 自行车横穿, 非机动车, 骑行者, bicycle, cyclist, crossing.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 8
BIKE_SPEED = 3
CROSSING_THRESHOLD = 18

behavior EgoBehavior():
    do FollowLaneBehavior(EGO_SPEED)

# CARLA bicycle blueprints ignore throttle and velocity commands. Move the real
# actor by a fixed distance each simulation tick after the existing trigger.
behavior CyclistBehavior():
    while (distance from self to ego) > CROSSING_THRESHOLD:
        wait
    while True:
        take OffsetAction(0 @ (BIKE_SPEED * 0.1))

lane = Uniform(*network.lanes)
spawn = OrientedPoint on lane.centerline,
    facing roadDirection
ego = Car following roadDirection from spawn for Range(-28, -20),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()
require (distance to intersection) > 60
cyclist = Bicycle at spawn offset by 3.5@0,
    with heading 90 deg relative to spawn.heading,
    with rolename "fdset_kinematic_bicycle",
    with behavior CyclistBehavior(),
    with regionContainedIn None
require (cyclist.position - ego.position).rotatedBy(-ego.heading).y > 12
