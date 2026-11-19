"""窄路路障通行 / roadblock narrows the drivable passage; ego approaches to yield or pass.
Keywords: 路障, 窄路, 道路障碍, roadblock, narrow passage, obstacle avoidance.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 7

behavior EgoBehavior():
    do FollowLaneBehavior(EGO_SPEED)

lane = Uniform(*network.lanes)
roadblockSpawn = OrientedPoint on lane.centerline,
    facing roadDirection
roadblock = Trash at roadblockSpawn offset by 1.85 @ 0
ego = Car following roadDirection from roadblockSpawn for Range(-26, -20),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()
require (distance to intersection) > 60
