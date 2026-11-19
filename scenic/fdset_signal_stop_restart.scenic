"""信号灯停车再起步 / signal stop and restart. The map light is frozen red, then changed to green.
Keywords: 红灯停车, 绿灯起步, 信号灯停车再起步, red light, green light, stop and go.
"""
param map = localPath('../maps/Town07.xodr')
param carla_map = 'Town07'
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"

monitor StopThenGreen:
    freezeTrafficLights()
    while not withinDistanceToTrafficLight(ego, 100):
        wait
    setClosestTrafficLightStatus(ego, "red")
    for _ in range(40):
        wait
    while True:
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
