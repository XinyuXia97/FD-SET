"""绿灯起步 / green-light start at a signalized intersection; no red-light claim.
Keywords: 绿灯起步, 信号灯起步, green light, traffic signal, stopped start.
"""
param map = localPath('../maps/Town07.xodr')
param carla_map = 'Town07'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"

monitor GreenLight:
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        wait

behavior EgoBehavior():
    do FollowLaneBehavior(8)

signalizedIntersections = filter(lambda i: any(s.isTrafficLight for s in i.signals), network.intersections)
intersection = Uniform(*signalizedIntersections)
startLane = Uniform(*intersection.incomingLanes)
ego = Car following roadDirection from startLane.centerline[-1] for Range(-18, -12),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()
