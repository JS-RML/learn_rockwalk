import rospy
import copy

import math
import numpy as np

from tf import transformations as tfms

from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, QuaternionStamped, TwistStamped
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion, Vector3

import Utilities


GRAVITY = 9.81 # also for energy calculation
APEX_HEIGHT = 1.35#1.35

#--------------------------- Object Size S--------------------------------------

# OBJECT_RADIUS = 0.15 # Base radius of (cone)
# CoM_X_OFFSET = 0.09
# CoM_Z_OFFSET = 0.40
# OBJECT_MASS = 1.0 # for energy calculation

#--------------------------- Object Size M--------------------------------------

# OBJECT_RADIUS = 0.225 # Base radius of (cone)
# CoM_X_OFFSET = 0.11
# CoM_Z_OFFSET = 0.36
# OBJECT_MASS = 1.4 # for energy calculation

#--------------------------- Object Size L--------------------------------------

OBJECT_RADIUS = 0.350 # Base radius of (cone)
CoM_X_OFFSET = 0.15
CoM_Z_OFFSET = 0.29
OBJECT_MASS = 1.8 # for energy calculation

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Check how you imu if fixed on the object relative to the body frame
# To see how body frame is defined take a look at the README.md file
IMU_BODYFRAME_QUATERNION = tfms.quaternion_from_euler(math.pi/2, 0, math.pi/2, 'rxyz')
# Read as: FROM IMU TO BODY FRAME QUATERNION

# Check how the imu (on the object) is initially oriented relative to the world frame
IMU_WORLDFRAME_QUATERNION = tfms.quaternion_from_euler(math.pi/2, 0, math.pi/2, 'rxyz')
# The above two definitions mean that initially the body frame is aligned with the world frame.
#----

class Kinematics:

    def __init__(self, sub_motion_shield, pub_kinematics):

        self._sub_motion_shield = sub_motion_shield
        self._pub_kinematics = pub_kinematics


        global CURRENT_POSITION, CURRENT_TIME

        CURRENT_POSITION = Vector3(0,0,0) # Begin position of the object at 0,0,0
        CURRENT_TIME = rospy.get_time()

        self.record_initial_object_configuration()


    def record_initial_object_configuration(self):

        rospy.loginfo("Bring the object to its initial configuration:")
        rospy.sleep(2)
        self._earth_initial_configuration_quaternion = self._sub_motion_shield._imu_quaternion
        rospy.loginfo("Initial IMU quaternion recorded")

    def compute_body_euler(self):

        earth_worldframe_quaternion = tfms.quaternion_multiply(self._earth_initial_configuration_quaternion,
                                                               IMU_WORLDFRAME_QUATERNION)

        earth_bodyframe_quaternion = tfms.quaternion_multiply(self._sub_motion_shield._imu_quaternion,
                                                              IMU_BODYFRAME_QUATERNION)

        self._world_bodyframe_quaternion = tfms.quaternion_multiply(tfms.quaternion_inverse(earth_worldframe_quaternion),
                                                                    earth_bodyframe_quaternion)

        world_bodyframe_rot = tfms.quaternion_matrix(self._world_bodyframe_quaternion)

        [psi, theta, phi] = Utilities.compute_euler(world_bodyframe_rot)

        self._body_euler = Vector3(psi, theta, phi) # added yaw offset here.

        self.compute_Q_frame() # compute Q frame

        #publish
        self._pub_kinematics._euler_publisher.publish(self._body_euler)

    def compute_Q_frame(self):
        # Q frame only tilts and yaws, but does not spin. Useful to determine ground contact
        rot_psi = tfms.rotation_matrix(self._body_euler.x, [0,0,1])
        init_rot = tfms.rotation_matrix(math.pi/2, [0,0,1])
        rot_theta = tfms.rotation_matrix(self._body_euler.y, [0,1,0])
        rot_phi = tfms.rotation_matrix(self._body_euler.z, [0,0,1])

        self._world_Q_rot = np.matmul(np.matmul(rot_psi, init_rot),rot_theta)

    def compute_body_twist(self):

        #-------------------------------
        #Angular velocity
        twist_vec = np.array([[np.radians(self._sub_motion_shield._imu_twist.x)],
                              [np.radians(self._sub_motion_shield._imu_twist.y)],
                              [np.radians(self._sub_motion_shield._imu_twist.z)],
                              [0]])

        imu_bodyframe_rot = tfms.quaternion_matrix(IMU_BODYFRAME_QUATERNION)
        angular_velocity_bodyframe = np.matmul(np.transpose(imu_bodyframe_rot),twist_vec)

        #-------------------------------
        #Translational Velocity from nonholonomic contact
        psi = self._body_euler.x
        theta = self._body_euler.y
        phi = self._body_euler.z

        psi_dot = angular_velocity_bodyframe[0,0]
        theta_dot = angular_velocity_bodyframe[1,0]
        phi_dot = angular_velocity_bodyframe[2,0]

        trans_vel_x = OBJECT_RADIUS*psi_dot*math.cos(psi)*math.cos(theta) - \
                      OBJECT_RADIUS*theta_dot*math.sin(psi)*math.sin(theta) + \
                      OBJECT_RADIUS*phi_dot*math.cos(psi)

        trans_vel_y = OBJECT_RADIUS*psi_dot*math.sin(psi)*math.cos(theta) - \
                      OBJECT_RADIUS*theta_dot*math.cos(psi)*math.sin(theta) + \
                      OBJECT_RADIUS*phi_dot*math.sin(psi)

        trans_vel_z = OBJECT_RADIUS*theta_dot*math.cos(theta)

        #-------------------------------

        self._body_twist = TwistStamped()
        self._body_twist.header.stamp = rospy.Time.now()
        self._body_twist.twist.angular = Vector3(psi_dot, theta_dot, phi_dot) # in rad/s
        self._body_twist.twist.linear = Vector3(trans_vel_x, trans_vel_y, trans_vel_z)

        #publish
        self._pub_kinematics._twist_publisher.publish(self._body_twist)

    def compute_object_pose(self):
        """
        Position: Base (disk) center coordinates (x_D, y_D, z_D)
        Orientation: Quaternion representing body frame relative to the world frame
        """
        global CURRENT_POSITION, CURRENT_TIME

        [CURRENT_POSITION, CURRENT_TIME] = self.integrate_linear_velocities(CURRENT_POSITION,
                                                                            CURRENT_TIME)

        self._object_pose = PoseStamped()
        self._object_pose.header.stamp = rospy.Time.now()
        self._object_pose.header.frame_id = "world"
        self._object_pose.pose.position = Point(self._disk_center_position.x,
                                                self._disk_center_position.y,
                                                self._disk_center_position.z)


        self._object_pose.pose.orientation = Quaternion(self._world_bodyframe_quaternion[0],
                                                        self._world_bodyframe_quaternion[1],
                                                        self._world_bodyframe_quaternion[2],
                                                        self._world_bodyframe_quaternion[3])


        #publish
        self._pub_kinematics._object_pose_publisher.publish(self._object_pose)

    def integrate_linear_velocities(self, current_position, current_time):
        """Base of the cone modeled as a circular disk"""

        new_time = self._body_twist.header.stamp.to_sec()

        position_change_x = self._body_twist.twist.linear.x * (new_time - current_time)
        position_change_y = self._body_twist.twist.linear.y * (new_time - current_time)

        self._disk_center_position = Vector3()
        self._disk_center_position.x = current_position.x + position_change_x
        self._disk_center_position.y = current_position.y + position_change_y
        self._disk_center_position.z = OBJECT_RADIUS*math.sin(self._body_euler.y)

        return self._disk_center_position, new_time

    def compute_contact_coordinates(self):
        """Trace of ground contact as the object rolls without slipping"""

        ground_contact_vector = np.matmul(self._world_Q_rot, np.array([[OBJECT_RADIUS],[0],[0],[0]]))

        contact_position_x = self._disk_center_position.x + ground_contact_vector[0,0]
        contact_position_y = self._disk_center_position.y + ground_contact_vector[1,0]

        self._ground_contact = Vector3()
        self._ground_contact.x = contact_position_x
        self._ground_contact.y = contact_position_y
        self._ground_contact.z = 0

        #publish
        self._pub_kinematics._ground_contact_publisher.publish(self._ground_contact)

    def compute_apex_coordinates(self):
        world_bodyframe_rot = tfms.quaternion_matrix(self._world_bodyframe_quaternion)
        disk_center_apex_vector = np.matmul(world_bodyframe_rot, np.array([[OBJECT_RADIUS],[0],[APEX_HEIGHT],[0]]))

        self._apex_position = Vector3()
        self._apex_position.x = self._disk_center_position.x + disk_center_apex_vector[0,0]
        self._apex_position.y = self._disk_center_position.y + disk_center_apex_vector[1,0]
        self._apex_position.z = self._disk_center_position.z + disk_center_apex_vector[2,0]

        self._pub_kinematics._apex_coordinates_publisher.publish(self._apex_position)


    def compute_apex_velocity(self):

        # D:= disk center

        # recall Q frame does not roll with the disk. Only tilts and yaws
        Q_bodyframe_rot = tfms.rotation_matrix(self._body_euler.z, [0,0,1]);
        Q_bodyframe_omega_SPATIAL = [0,0,self._body_twist.twist.angular.z]

        D_apex_vector_Q = np.matmul(Q_bodyframe_rot, np.array([[OBJECT_RADIUS],[0],[APEX_HEIGHT],[0]])) # vector from disk center to apex in Q frame

        D_apex_velocity_Q = np.cross(Q_bodyframe_omega_SPATIAL, [D_apex_vector_Q[0,0],D_apex_vector_Q[1,0],D_apex_vector_Q[2,0]]) # velocity of apex relative to disk center expressed in Q frame

        world_Q_omega_BODY = [-self._body_twist.twist.angular.x*math.sin(self._body_euler.y),
                               self._body_twist.twist.angular.y,
                               self._body_twist.twist.angular.x*math.cos(self._body_euler.y)]

        apex_velocity_Q = np.cross(world_Q_omega_BODY, [D_apex_vector_Q[0,0],D_apex_vector_Q[1,0],D_apex_vector_Q[2,0]]) + D_apex_velocity_Q

        D_velocity_world = np.array([[self._body_twist.twist.linear.x],[self._body_twist.twist.linear.y],[self._body_twist.twist.linear.z],[0]])

        apex_velocity_world = D_velocity_world + np.matmul(self._world_Q_rot, np.array([[apex_velocity_Q[0]],[apex_velocity_Q[1]],[apex_velocity_Q[2]],[0]]))

        self._apex_velocity = Vector3()
        self._apex_velocity.x = apex_velocity_world[0,0]
        self._apex_velocity.y = apex_velocity_world[1,0]
        self._apex_velocity.z = apex_velocity_world[2,0]

        self._pub_kinematics._apex_velocity_publisher.publish(self._apex_velocity)

    def compute_CoM_coordinates(self):
        world_bodyframe_rot = tfms.quaternion_matrix(self._world_bodyframe_quaternion)
        disk_center_CoM_vector = np.matmul(world_bodyframe_rot, np.array([[CoM_X_OFFSET],[0],[CoM_Z_OFFSET],[0]]))

        self._CoM_position = Vector3()
        self._CoM_position.x = self._disk_center_position.x + disk_center_CoM_vector[0]
        self._CoM_position.y = self._disk_center_position.y + disk_center_CoM_vector[1]
        self._CoM_position.z = self._disk_center_position.z + disk_center_CoM_vector[2]

        self._pub_kinematics._CoM_coordinates_publisher.publish(self._CoM_position)

    def compute_CoM_velocity(self):

        # D:= disk center

        # recall Q frame does not roll with the disk. Only tilts and yaws
        Q_bodyframe_rot = tfms.rotation_matrix(self._body_euler.z, [0,0,1]);
        Q_bodyframe_omega_SPATIAL = [0,0,self._body_twist.twist.angular.z]

        D_CoM_vector_Q = np.matmul(Q_bodyframe_rot, np.array([[CoM_X_OFFSET],[0],[CoM_Z_OFFSET],[0]])) # vector from disk center to CoM in Q frame

        D_CoM_velocity_Q = np.cross(Q_bodyframe_omega_SPATIAL, [D_CoM_vector_Q[0,0],D_CoM_vector_Q[1,0],D_CoM_vector_Q[2,0]]) # velocity of CoM relative to disk center expressed in Q frame

        world_Q_omega_BODY = [-self._body_twist.twist.angular.x*math.sin(self._body_euler.y),
                               self._body_twist.twist.angular.y,
                               self._body_twist.twist.angular.x*math.cos(self._body_euler.y)]

        CoM_velocity_Q = np.cross(world_Q_omega_BODY, [D_CoM_vector_Q[0,0],D_CoM_vector_Q[1,0],D_CoM_vector_Q[2,0]]) + D_CoM_velocity_Q

        D_velocity_world = np.array([[self._body_twist.twist.linear.x],[self._body_twist.twist.linear.y],[self._body_twist.twist.linear.z],[0]])

        CoM_velocity_world = D_velocity_world + np.matmul(self._world_Q_rot, np.array([[CoM_velocity_Q[0]],[CoM_velocity_Q[1]],[CoM_velocity_Q[2]],[0]]))

        self._CoM_velocity = Vector3()
        self._CoM_velocity.x = CoM_velocity_world[0,0]
        self._CoM_velocity.y = CoM_velocity_world[1,0]
        self._CoM_velocity.z = CoM_velocity_world[2,0]

        self._pub_kinematics._CoM_velocity_publisher.publish(self._CoM_velocity)


    def compute_angular_velocity(self):

        q4 = self._body_euler.x
        q5 = self._body_euler.y
        dqdt4 = self._body_twist.twist.angular.x
        dqdt5 = self._body_twist.twist.angular.y
        dqdt6 = self._body_twist.twist.angular.z

        self._angular_velocity = Vector3()
        self._angular_velocity.x = -dqdt5*math.cos(q4) - dqdt6*math.sin(q4)*math.sin(q5)
        self._angular_velocity.y = dqdt6*math.cos(q4)*math.sin(q5) - dqdt5*math.sin(q4)
        self._angular_velocity.z = dqdt4 + dqdt6*math.cos(q5)

        self._pub_kinematics._angular_velocity_publisher.publish(self._angular_velocity)

    def compute_energies(self):

        CoM_velocity = [self._CoM_velocity.x, self._CoM_velocity.y, self._CoM_velocity.z]

        self._kinetic_energy = Float64()

        self._kinetic_energy.data = 0.5*OBJECT_MASS*np.dot(CoM_velocity, CoM_velocity)
        self._pub_kinematics._kinetic_energy_publisher.publish(self._kinetic_energy)

        self._potential_energy = Float64()

        self._potential_energy.data = OBJECT_MASS*GRAVITY*self._CoM_position.z
        self._pub_kinematics._potential_energy_publisher.publish(self._potential_energy)

        self._total_energy = Float64()
        self._total_energy.data = self._kinetic_energy.data + self._potential_energy.data
        self._pub_kinematics._total_energy_publisher.publish(self._total_energy)


    def compute_min_nutation(self, prev_min_nutation, prev_detect_mean_pos_nutation):
        if abs(self._body_twist.twist.angular.z) > 0.40 and abs(self._body_euler.z) < 0.05 and prev_detect_mean_pos_nutation==True:

            min_nutation = Float64(self._body_euler.y)
            detect_mean_pos_nutation = False

        else:
            min_nutation = copy.deepcopy(prev_min_nutation)
            detect_mean_pos_nutation = True

        self._pub_kinematics._min_nutation_publisher.publish(min_nutation)

        return min_nutation, detect_mean_pos_nutation



    def compute_pke_dke_dkedt(self, prev_pke, prev_dke, prev_dkedt, prev_t, prev_detect_mean_pos):

        # compute_peak_kinetic_energy
        if abs(self._body_twist.twist.angular.z) > 0.40 and abs(self._body_euler.z) < 0.05 and prev_detect_mean_pos==True:

            pke = copy.deepcopy(self._kinetic_energy)
            dke = Float64(pke.data - prev_pke.data)
            diff_dke = dke.data - prev_dke.data

            # compute_time_derivative of peak kinetic energy
            curr_t = rospy.get_time()
            dt = curr_t - prev_t
            dkedt = Float64(diff_dke/dt)

            detect_mean_pos = False


        elif self._kinetic_energy.data < 5e-9:
            pke = Float64(0)
            dke = Float64(0)
            dkedt = Float64(0)
            curr_t = prev_t
            detect_mean_pos = prev_detect_mean_pos

        elif abs(self._body_twist.twist.angular.z) < 0.50 and prev_detect_mean_pos==False:
            pke =  copy.deepcopy(prev_pke)
            dke = copy.deepcopy(prev_dke)
            dkedt = copy.deepcopy(prev_dkedt)
            curr_t = prev_t
            detect_mean_pos = True

        else:
            pke =  copy.deepcopy(prev_pke)
            dke = copy.deepcopy(prev_dke)
            dkedt = copy.deepcopy(prev_dkedt)
            curr_t = prev_t
            detect_mean_pos = prev_detect_mean_pos


        self._pub_kinematics._pke_publisher.publish(pke)
        self._pub_kinematics._dke_publisher.publish(dke)
        self._pub_kinematics._dkedt_publisher.publish(dkedt)

        return pke, dke, dkedt, curr_t, detect_mean_pos

    def compute_average_forward_speed(self, previous_disk_center_position, previous_average_forward_speed,
                                      previous_d_average_forward_speed_dt, previous_time_2, prev_detect_mean_pos_2):

        # compute_peak_kinetic_energy
        if abs(self._body_twist.twist.angular.z) > 0.40 and abs(self._body_euler.z) < np.radians(2) and prev_detect_mean_pos_2==True:

            # compute average speed
            current_time = rospy.get_time()
            dt = current_time - previous_time_2
            delta_x = self._disk_center_position.x - previous_disk_center_position.x
            delta_y = self._disk_center_position.y - previous_disk_center_position.y
            # net_distance = math.sqrt(delta_x**2 + delta_y**2)
            average_forward_speed = Float64(delta_x/dt)
            average_forward_speed_diff = average_forward_speed.data - previous_average_forward_speed.data
            d_average_forward_speed_dt = Float64(average_forward_speed_diff/dt)
            disk_center_position = copy.deepcopy(self._disk_center_position)

            detect_mean_pos_2 = False

        elif self._kinetic_energy.data < 1e-8:

            disk_center_position = copy.deepcopy(self._disk_center_position)
            average_forward_speed = Float64(0)
            d_average_forward_speed_dt = Float64(0)
            current_time = previous_time_2
            detect_mean_pos_2 = prev_detect_mean_pos_2

        elif abs(self._body_twist.twist.angular.z) < 0.50 and prev_detect_mean_pos_2==False:

            disk_center_position = copy.deepcopy(previous_disk_center_position)
            average_forward_speed = copy.deepcopy(previous_average_forward_speed)
            d_average_forward_speed_dt = copy.deepcopy(previous_d_average_forward_speed_dt)
            current_time = previous_time_2
            detect_mean_pos_2 = True

        else:

            disk_center_position = copy.deepcopy(previous_disk_center_position)
            average_forward_speed = copy.deepcopy(previous_average_forward_speed)
            d_average_forward_speed_dt = copy.deepcopy(previous_d_average_forward_speed_dt)
            current_time = previous_time_2
            detect_mean_pos_2 = prev_detect_mean_pos_2

        self._pub_kinematics._average_forward_speed_publisher.publish(average_forward_speed)
        self._pub_kinematics._average_forward_speed_derivative_publisher.publish(d_average_forward_speed_dt)



        return disk_center_position, average_forward_speed, d_average_forward_speed_dt, current_time, detect_mean_pos_2





    def compute_next_waypoint(self, previous_waypoints, rockwalk_step):

        precess_angle = math.radians(6)
        angle_range = np.linspace(0,precess_angle,10)

        if self._body_twist.twist.angular.z > 0.3 and self._body_twist.twist.angular.z < 1.6 \
            and self._body_euler.z > np.radians(18) and (rockwalk_step % 2) == 0:


            vec_GA = np.array([[self._apex_position.x - self._ground_contact.x],
                               [self._apex_position.y - self._ground_contact.y],
                               [self._apex_position.z - self._ground_contact.z],
                               [0]])

            next_waypoints = PoseArray()


            for idx in range(len(angle_range)):
                rot_z = tfms.rotation_matrix(-angle_range[idx], [0,0,1])
                vec_GA_prime = np.matmul(rot_z, vec_GA)

                next_pose = Pose(Point(self._ground_contact.x + vec_GA_prime[0,0],
                                       self._ground_contact.y + vec_GA_prime[1,0],
                                       self._ground_contact.z + vec_GA_prime[2,0]),
                                 Quaternion(0,0,0,1))

                next_waypoints.poses.append(next_pose)

            self._pub_kinematics._next_waypoint_publisher.publish(next_waypoints)

            rockwalk_step += 1

            return next_waypoints, rockwalk_step

        elif self._body_twist.twist.angular.z < -0.3  and self._body_twist.twist.angular.z > -1.6 \
            and self._body_euler.z < np.radians(-18) and (rockwalk_step % 2) == 1:

            vec_GA = np.array([[self._apex_position.x - self._ground_contact.x],
                               [self._apex_position.y - self._ground_contact.y],
                               [self._apex_position.z - self._ground_contact.z],
                               [0]])

            next_waypoints = PoseArray()

            for idx in range(len(angle_range)):
                rot_z = tfms.rotation_matrix(angle_range[idx], [0,0,1])
                vec_GA_prime = np.matmul(rot_z, vec_GA)

                next_pose = Pose(Point(self._ground_contact.x + vec_GA_prime[0,0],
                                       self._ground_contact.y + vec_GA_prime[1,0],
                                       self._ground_contact.z + vec_GA_prime[2,0]),
                                 Quaternion(0,0,0,1))

                next_waypoints.poses.append(next_pose)


            self._pub_kinematics._next_waypoint_publisher.publish(next_waypoints)

            rockwalk_step += 1

            return next_waypoints, rockwalk_step


        else:
            self._pub_kinematics._next_waypoint_publisher.publish(previous_waypoints)
            return previous_waypoints, rockwalk_step
