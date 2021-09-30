
import rospy
import copy
import math
import tf
import numpy as np

from tf import transformations as tfms

from geometry_msgs.msg import Point, Quaternion, Pose, PoseArray, Vector3, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from nav_msgs.msg import Path

OBJECT_RADIUS = 0.351
# FROM BODY FRAME TO MARKER FRAME
BODY_MARKERFRAME_QUATERNION = tfms.quaternion_from_euler(math.pi/2, math.pi/2, 0, 'rxyz')

BODY_MARKERFRAME_POSITION = np.array([[-OBJECT_RADIUS],[-OBJECT_RADIUS],[0],[0]])


class KinematicsVisualization:

    def __init__(self, pub_visualization, sub_kinematics):

        self._pub_visualization = pub_visualization
        self._sub_kinematics = sub_kinematics

        # store sequence of data for visualization
        self._ground_contact_points = []
        self._apex_points = []
        self._CoM_points = []

        self._waypoint_path = Path()
        self._waypoint_path.header.frame_id = "world"



    def publish_object_marker(self):


        world_bodyframe_quaternion = np.array([self._sub_kinematics._object_pose.pose.orientation.x,
                                               self._sub_kinematics._object_pose.pose.orientation.y,
                                               self._sub_kinematics._object_pose.pose.orientation.z,
                                               self._sub_kinematics._object_pose.pose.orientation.w])


        world_markerframe_quaternion = tfms.quaternion_multiply(world_bodyframe_quaternion,BODY_MARKERFRAME_QUATERNION)

        world_bodyframe_rot = tfms.quaternion_matrix(world_bodyframe_quaternion)
        world_diskcenter_marker_vec = np.matmul(world_bodyframe_rot, BODY_MARKERFRAME_POSITION)

        marker_pose = Pose()
        marker_pose.position = Point(self._sub_kinematics._object_pose.pose.position.x + world_diskcenter_marker_vec[0],
                                     self._sub_kinematics._object_pose.pose.position.y + world_diskcenter_marker_vec[1],
                                     self._sub_kinematics._object_pose.pose.position.z + world_diskcenter_marker_vec[2])

        marker_pose.orientation = Quaternion(world_markerframe_quaternion[0],
                                             world_markerframe_quaternion[1],
                                             world_markerframe_quaternion[2],
                                             world_markerframe_quaternion[3])

        marker = Marker(
                    type=Marker.MESH_RESOURCE,
                    action=Marker.ADD,
                    id=0,
                    pose = marker_pose,
                    scale=Vector3(0.001, .001, .001),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(.70,.70,.70 ,1),
                    mesh_resource="package://rockwalk_kinematics/object_model/rockwalk_cone.dae"
                    )

        self._pub_visualization._object_marker_publisher.publish(marker)


    def publish_kong_tcp_velocity_marker(self):

        # not really velocity. just difference in position
        vector_start_point = copy.deepcopy(self._apex_position)


        kong_tcp_position_diff = Point(self._sub_kinematics._kong_tcp_pose.position.x-(-0.15),
                                       self._sub_kinematics._kong_tcp_pose.position.y-(-0.70),
                                       self._sub_kinematics._kong_tcp_pose.position.z-(0.48))


        vector_end_point = Point(vector_start_point.x + 2*kong_tcp_position_diff.x,
                                 vector_start_point.y + 2*kong_tcp_position_diff.y,
                                 vector_start_point.z + 2*kong_tcp_position_diff.z)

        marker = Marker(
                    type=Marker.ARROW,
                    action=Marker.ADD,
                    id=31,
                    points = [vector_start_point, vector_end_point],
                    scale=Vector3(0.015, .030, 0.030),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(0,0,0.545,1)
                    )

        self._pub_visualization._kong_tcp_velocity_marker_publisher.publish(marker)

    def publish_contact_path_marker(self):


        self._ground_contact_point = Point(self._sub_kinematics._ground_contact.x,
                                           self._sub_kinematics._ground_contact.y,
                                           self._sub_kinematics._ground_contact.z)

        self._ground_contact_points.append(self._ground_contact_point)


        marker = Marker(
                    type=Marker.LINE_STRIP,#Marker.SPHERE_LIST,
                    action=Marker.ADD,
                    id=1,
                    points = self._ground_contact_points,
                    scale=Vector3(0.01, .01, .01),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(.63,.32,.18 ,1)
                    )

        self._pub_visualization._contact_path_marker_publisher.publish(marker)


    def publish_apex_position_marker(self):

        self._apex_position = Point(self._sub_kinematics._apex_coordinates.x,
                                    self._sub_kinematics._apex_coordinates.y,
                                    self._sub_kinematics._apex_coordinates.z)


        marker = Marker(
                    type=Marker.SPHERE,
                    action=Marker.ADD,
                    id=21,
                    pose = Pose(self._apex_position,Quaternion(0,0,0,1)),
                    scale=Vector3(0.020, .020, .020),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(0,0,1,1)
                    )

        self._pub_visualization._apex_position_marker_publisher.publish(marker)



    def publish_apex_path_marker(self):



        self._apex_points.append(self._apex_position)

        marker = Marker(
                    type=Marker.LINE_STRIP,
                    action=Marker.ADD,
                    id=2,
                    points = self._apex_points,
                    scale=Vector3(0.005, .01, .01),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(0.25,0.41,0.88,1)
                    )

        self._pub_visualization._apex_path_marker_publisher.publish(marker)


    def publish_apex_velocity_marker(self):


        self._apex_velocity = Point(self._sub_kinematics._apex_velocity.x,
                                    self._sub_kinematics._apex_velocity.y,
                                    self._sub_kinematics._apex_velocity.z)

        vector_start_point = copy.deepcopy(self._apex_position)
        vector_end_point = Point(vector_start_point.x + 0.4*self._apex_velocity.x,
                                 vector_start_point.y + 0.4*self._apex_velocity.y,
                                 vector_start_point.z + 0.4*self._apex_velocity.z)

        marker = Marker(
                    type=Marker.ARROW,
                    action=Marker.ADD,
                    id=3,
                    points = [vector_start_point, vector_end_point],
                    scale=Vector3(0.015, .030, 0.030),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(0,0,0.545,1)
                    )

        self._pub_visualization._apex_velocity_marker_publisher.publish(marker)

    def publish_CoM_marker(self):

        self._CoM_position = Point(self._sub_kinematics._CoM_coordinates.x,
                                   self._sub_kinematics._CoM_coordinates.y,
                                   self._sub_kinematics._CoM_coordinates.z)


        marker = Marker(
                    type=Marker.SPHERE,
                    action=Marker.ADD,
                    id=4,
                    pose = Pose(self._CoM_position, Quaternion(0,0,0,1)),
                    scale=Vector3(0.04, .04, .04),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(.86,.08,.24,1)
                    )

        self._pub_visualization._CoM_marker_publisher.publish(marker)

    def publish_CoM_path_marker(self):

        self._CoM_points.append(self._CoM_position)

        marker = Marker(
                    type=Marker.LINE_STRIP,
                    action=Marker.ADD,
                    id=5,
                    points = self._CoM_points,
                    scale=Vector3(0.005, .01, .01),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(1,.27,0,1)
                    )

        self._pub_visualization._CoM_path_marker_publisher.publish(marker)

    def publish_CoM_velocity_marker(self):


        self._CoM_velocity = Point(self._sub_kinematics._CoM_velocity.x,
                                   self._sub_kinematics._CoM_velocity.y,
                                   self._sub_kinematics._CoM_velocity.z)

        vector_start_point = copy.deepcopy(self._CoM_position)
        vector_end_point = Point(vector_start_point.x + 0.4*self._CoM_velocity.x,
                                 vector_start_point.y + 0.4*self._CoM_velocity.y,
                                 vector_start_point.z + 0.4*self._CoM_velocity.z)

        marker = Marker(
                    type=Marker.ARROW,
                    action=Marker.ADD,
                    id=6,
                    points = [vector_start_point, vector_end_point],
                    scale=Vector3(0.015, .030, 0.030),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(.86,.08,.24,1)
                    )

        self._pub_visualization._CoM_velocity_marker_publisher.publish(marker)


    def publish_angular_velocity_marker(self):

        self._angular_velocity = Point(self._sub_kinematics._angular_velocity.x,
                                       self._sub_kinematics._angular_velocity.y,
                                       self._sub_kinematics._angular_velocity.z)

        vector_start_point = copy.deepcopy(self._ground_contact_point)
        vector_end_point = Point(vector_start_point.x + 0.4*self._angular_velocity.x,
                                 vector_start_point.y + 0.4*self._angular_velocity.y,
                                 vector_start_point.z + 0.4*self._angular_velocity.z)

        marker = Marker(
                    type=Marker.ARROW,
                    action=Marker.ADD,
                    id=7,
                    points = [vector_start_point, vector_end_point],
                    scale=Vector3(0.015, .030, 0.030),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(.86,.38,.34,1)
                    )

        self._pub_visualization._angular_velocity_marker_publisher.publish(marker)


    def publish_apex_plane_marker(self):

        plane_center = copy.deepcopy(self._apex_position)

        marker = Marker(
                    type=Marker.CUBE,
                    action=Marker.ADD,
                    id=8,
                    pose = Pose(plane_center,Quaternion(0,0,0,1)),
                    scale=Vector3(1.0, 0.5, 0.001),
                    header=Header(frame_id="world"),
                    color=ColorRGBA(0.1,0.1,0.1,0.3)
                    )

        self._pub_visualization._apex_plane_marker_publisher.publish(marker)



    def publish_next_waypoint_marker(self):

        end_waypoint = copy.deepcopy(self._sub_kinematics._next_waypoint.poses[-1])

        marker = Marker(
                        type=Marker.SPHERE,
                        action=Marker.ADD,
                        id=9,
                        pose = end_waypoint,
                        scale=Vector3(0.020, .020, .020),
                        header=Header(frame_id="world"),
                        color=ColorRGBA(0,0.8,0.2,1)
                        )

        self._pub_visualization._next_waypoint_marker_publisher.publish(marker)



    def publish_next_waypoint_path_marker(self):

        waypoints = []

        for pose in self._sub_kinematics._next_waypoint.poses:
            waypoints.append(pose.position)

        marker = Marker(
                        type=Marker.LINE_STRIP,
                        action=Marker.ADD,
                        id=21,
                        points = waypoints,
                        scale=Vector3(0.005, .01, .01),
                        header=Header(frame_id="world"),
                        color=ColorRGBA(0,1,1.0,1)
                        )

        self._pub_visualization._next_waypoint_path_marker_publisher.publish(marker)

    def publish_waypoint_path(self):

        for pose in self._sub_kinematics._next_waypoint.poses:
            ps = PoseStamped()
            ps.header.frame_id = "world"
            # ps.header.stamp = rospy.Time.now()
            ps.pose = copy.deepcopy(pose)

            self._waypoint_path.poses.append(ps)

        self._pub_visualization._waypoint_path_publisher.publish(self._waypoint_path)
