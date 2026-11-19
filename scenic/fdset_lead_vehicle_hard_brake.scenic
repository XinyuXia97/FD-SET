"""同道前车急刹 / same-lane lead vehicle hard braking; ego follows at short headway.
Keywords: 前车急刹, 同道跟车, 紧急制动, lead vehicle, hard brake, car following.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior EgoBehavior():
    do FollowLaneBehavior(9)

behavior LeadBrakeBehavior():
    try:
        do FollowLaneBehavior(7)
    interrupt when withinDistanceToAnyCars(self, 12):
        take SetBrakeAction(1.0)

lane = Uniform(*network.lanes)
lead = Car on lane.centerline,
    with behavior LeadBrakeBehavior()
ego = Car following roadDirection from lead for Range(-14, -9),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()
require (distance to intersection) > 50
