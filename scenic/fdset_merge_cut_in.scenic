"""相邻车道并线切入 / adjacent-lane vehicle merges into ego lane.
Keywords: 并线切入, 加塞, lane merge, cut-in, merging vehicle.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')
param carla_map = 'Town_Safebench_Light'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 8

behavior EgoBehavior():
    do FollowLaneBehavior(EGO_SPEED)

behavior CutInBehavior(targetLane):
    do LaneChangeBehavior(laneSectionToSwitch=targetLane, target_speed=7)

sectionsWithRightLane = []
for lane in network.lanes:
    for section in lane.sections:
        if section._laneToRight is not None:
            sectionsWithRightLane.append(section)

sourceSection = Uniform(*sectionsWithRightLane)
neighborSection = sourceSection._laneToRight
ego = Car on sourceSection.centerline,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()
mergeCar = Car on neighborSection.centerline,
    with behavior CutInBehavior(sourceSection)
require (distance from ego to mergeCar) >= 6
require (distance from ego to mergeCar) <= 20
require (mergeCar.position - ego.position).rotatedBy(-ego.heading).y > 6
require (distance to intersection) > 50
