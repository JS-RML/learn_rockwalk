import rospy

from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3, PoseStamped, PoseArray, TwistStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path


class KinematicsPublishers:

    def __init__(self):
        self.initialize_publishers()

    def initialize_publishers(self):
        rospy.loginfo("Initializing the following topics:")
        rospy.loginfo("1. Body euler @body_euler")
        rospy.loginfo("2. Body twist @body_twist")
        rospy.loginfo("3. Object pose @object_pose")
        rospy.loginfo("4. Ground contact @ground_contact")
        rospy.loginfo("5. Apex coordinates @apex_coordinates")
        rospy.loginfo("6. CoM coordinates @CoM_coordinates")
        rospy.loginfo("7. CoM velocity @CoM_velocity")
        rospy.loginfo("8. Next Waypoint @next_waypoint")

        self._euler_publisher = rospy.Publisher('body_euler', Vector3, queue_size=10)
        self._twist_publisher = rospy.Publisher('body_twist', TwistStamped, queue_size=10)
        self._object_pose_publisher = rospy.Publisher('object_pose', PoseStamped, queue_size=10)
        self._ground_contact_publisher = rospy.Publisher('ground_contact', Vector3, queue_size=10)
        self._apex_coordinates_publisher = rospy.Publisher('apex_coordinates', Vector3, queue_size=10)
        self._apex_velocity_publisher = rospy.Publisher('apex_velocity', Vector3, queue_size=10)
        self._CoM_coordinates_publisher = rospy.Publisher('CoM_coordinates', Vector3, queue_size=10)
        self._CoM_velocity_publisher = rospy.Publisher('CoM_velocity', Vector3, queue_size=10)
        self._angular_velocity_publisher = rospy.Publisher('angular_velocity', Vector3, queue_size=10)
        self._kinetic_energy_publisher = rospy.Publisher('kinetic_energy', Float64, queue_size=10)
        self._potential_energy_publisher = rospy.Publisher('potential_energy', Float64, queue_size=10)
        self._total_energy_publisher = rospy.Publisher('total_energy', Float64, queue_size=10)

        self._min_nutation_publisher = rospy.Publisher('min_nutation', Float64, queue_size=10)


        self._pke_publisher = rospy.Publisher('pke', Float64, queue_size=10)
        self._dke_publisher = rospy.Publisher('dke', Float64, queue_size=10)
        self._dkedt_publisher = rospy.Publisher('dkedt', Float64, queue_size=10)

        self._average_forward_speed_publisher = rospy.Publisher('average_forward_speed', Float64, queue_size=10)
        self._average_forward_speed_derivative_publisher = rospy.Publisher('average_forward_speed_derivative', Float64, queue_size=10)
        self._next_waypoint_publisher = rospy.Publisher('next_waypoint', PoseArray, queue_size=10)



class VisualizationPublishers:

    def __init__(self):
        self.initialize_publishers()

    def initialize_publishers(self):

        self._object_marker_publisher = rospy.Publisher('object_marker', Marker, queue_size=10)
        self._kong_tcp_velocity_marker_publisher = rospy.Publisher('kong_tcp_velocity_marker', Marker, queue_size=10)
        self._contact_path_marker_publisher = rospy.Publisher('contact_path_marker', Marker, queue_size=10)
        self._apex_position_marker_publisher = rospy.Publisher('apex_position_marker', Marker, queue_size=10)
        self._apex_path_marker_publisher = rospy.Publisher('apex_path_marker', Marker, queue_size=10)
        self._apex_velocity_marker_publisher = rospy.Publisher('apex_velocity_marker', Marker, queue_size=10)
        self._CoM_marker_publisher = rospy.Publisher('CoM_marker', Marker, queue_size=10)
        self._CoM_path_marker_publisher = rospy.Publisher('CoM_path_marker', Marker, queue_size=10)
        self._CoM_velocity_marker_publisher = rospy.Publisher('CoM_velocity_marker', Marker, queue_size=10)
        self._angular_velocity_marker_publisher = rospy.Publisher('angular_velocity_marker', Marker, queue_size=10)
        self._apex_plane_marker_publisher = rospy.Publisher('apex_plane_marker', Marker, queue_size=10)
        self._next_waypoint_marker_publisher = rospy.Publisher('next_waypoint_marker', Marker, queue_size=10)
        self._waypoint_path_publisher = rospy.Publisher('waypoint_path', Path, queue_size=10)
        self._next_waypoint_path_marker_publisher = rospy.Publisher('next_waypoint_path_marker', Marker, queue_size=10)
