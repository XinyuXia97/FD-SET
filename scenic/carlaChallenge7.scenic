"""Traffic Scenario 07: crossing traffic while ego drives straight through an intersection.
The original template does not control any traffic light, so it is not a red-light violation.
"""
param map = localPath('../maps/Town_Safebench_Light.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town_Safebench_Light'
model scenic.domains.driving.model

DELAY_TIME_1 = 1 # the delay time for ego
DELAY_TIME_2 = 40 # the delay time for the slow car
FOLLOWING_DISTANCE = 13 # normally 10, 40 when DELAY_TIME is 25, 50 to prevent collisions

DISTANCE_TO_INTERSECTION1 = Uniform(27, 32) * -1
DISTANCE_TO_INTERSECTION2 = Uniform(0, 5) * -1
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0


behavior CrossingCarBehavior(trajectory):
	while True:
		do FollowTrajectoryBehavior(trajectory = trajectory)
		do FollowLaneBehavior(10)
        
behavior EgoBehavior(trajectory):
	
	try:
		do FollowTrajectoryBehavior(trajectory=trajectory)
	interrupt when withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
		take SetBrakeAction(BRAKE_INTENSITY)


spawnAreas = []
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = Uniform(*fourWayIntersection)

startLane = Uniform(*intersec.incomingLanes)
straight_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, startLane.maneuvers)
straight_maneuver = Uniform(*straight_maneuvers)
param fdset_route_goal = straight_maneuver.endLane.centerline[-1]
ego_trajectory = [straight_maneuver.startLane, straight_maneuver.connectingLane, straight_maneuver.endLane]

conflicting_straight_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, straight_maneuver.conflictingManeuvers)
csm = Uniform(*conflicting_straight_maneuvers)
crossing_startLane = csm.startLane
crossing_car_trajectory = [csm.startLane, csm.connectingLane, csm.endLane]

ego_spwPt = startLane.centerline[-1]
csm_spwPt = crossing_startLane.centerline[-1]

ego = Car following roadDirection from ego_spwPt for DISTANCE_TO_INTERSECTION1,
		with behavior EgoBehavior(trajectory = ego_trajectory)

crossing_car = Car following roadDirection from csm_spwPt for DISTANCE_TO_INTERSECTION2,
				with behavior CrossingCarBehavior(crossing_car_trajectory)
