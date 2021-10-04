import urx
import rospy
import math3d as m3d


class RobotSetup:

    def __init__(self, robot_ip):

        self._robot = urx.Robot(robot_ip, use_rt=True)
        rospy.sleep(1)

    def get_robot(self):
        return self._robot

    def set_home_configuration(self):

        rospy.loginfo("Moving robot arm to home configuration")
        pose = m3d.Transform()
        pose.pos = m3d.Vector([-0.45, -0.8, 0.4])
        pose.orient = m3d.Orientation([0, 1, 0,
                                       1, 0, 0,
                                       0, 0, -1])

        self._robot.set_pose(pose, acc=0.05, vel=0.05, wait=False)
        rospy.sleep(0.5)
