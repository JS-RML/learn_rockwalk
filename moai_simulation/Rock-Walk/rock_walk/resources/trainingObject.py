import pybullet as bullet
import numpy as np
import trimesh

import os
from rock_walk.resources.utils import *

YAW_OFFSET = np.pi #0


class TrainObject:
    def __init__(self, clientID):
        self.clientID = clientID

    def load_model_from_urdf(self, yaw_spawn):
        self._yaw_spawn = yaw_spawn + YAW_OFFSET

        f_name1 = os.path.join(os.path.dirname(__file__),'models/auto_object.urdf')


        orientation = bullet.getQuaternionFromEuler([0,0,self._yaw_spawn],physicsClientId=self.clientID)
        self.objectID = bullet.loadURDF(fileName=f_name1,
                                      basePosition=[0, 0, 3],
                                      baseOrientation=orientation,#([0,0,np.pi/2]),
                                      useFixedBase=1,
                                      # flags=bullet.URDF_USE_INERTIA_FROM_FILE, #problematic
                                      physicsClientId=self.clientID)
    def load_texutre(self):
        f_name2 = os.path.join(os.path.dirname(__file__), 'models/texture/tex_0.jpg')
        texID = bullet.loadTexture(f_name2)
        bullet.changeVisualShape(self.objectID, 5, textureUniqueId=texID, physicsClientId=self.clientID)

    def get_ids(self):
        return self.objectID, self.clientID

    def get_joint_info(self, idx):
        print(bullet.getJointInfo(self.objectID, idx, physicsClientId=self.clientID))

    def get_dynamics_info(self):
        print(bullet.getDynamicsInfo(self.objectID, 5, physicsClientId=self.clientID))

    def set_lateral_friction(self, value):
        bullet.changeDynamics(self.objectID, 5, lateralFriction=value, physicsClientId=self.clientID)

    def apply_action(self, action):


        action_rot = R.from_euler('z', -self._yaw_spawn + YAW_OFFSET).as_matrix()
        action = np.matmul(action_rot, np.array([action[0], action[1], 0]))


        bullet.setJointMotorControl2(self.objectID,
                                     jointIndex=0,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     targetVelocity=action[0],
                                     physicsClientId=self.clientID)

        bullet.setJointMotorControl2(self.objectID,
                                     jointIndex=1,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     targetVelocity=action[1],
                                     physicsClientId=self.clientID)

        bullet.setJointMotorControl2(self.objectID,
                                     jointIndex=2,
                                     controlMode=bullet.VELOCITY_CONTROL,
                                     targetVelocity=action[2],
                                     force=0,
                                     physicsClientId=self.clientID)


    def apply_perturbation_force(self, force_vector, pos_vector):
        bullet.applyExternalForce(self.objectID,
                                  linkIndex=5,
                                  forceObj=force_vector,
                                  posObj=pos_vector,
                                  flags=bullet.LINK_FRAME,
                                  physicsClientId=self.clientID)


    def get_control_point_position(self):
        lin_pos_world, quat_world = bullet.getLinkState(self.objectID,linkIndex=4, physicsClientId=self.clientID)[0:2]

        return lin_pos_world

    def get_control_point_velocity(self):
        lin_vel_world, ang_vel_world = bullet.getLinkState(self.objectID,linkIndex=4,computeLinkVelocity=1,
                                                                     physicsClientId=self.clientID)[-2:]

        return lin_vel_world




    def get_observation(self):
        lin_pos_base_world, quat_base_world = bullet.getLinkState(self.objectID,linkIndex=5,physicsClientId=self.clientID)[0:2]
        lin_vel_base_world, ang_vel_base_world = bullet.getLinkState(self.objectID,linkIndex=5,computeLinkVelocity=1,
                                                                     physicsClientId=self.clientID)[-2:]

        rot_base_world = bullet.getMatrixFromQuaternion(quat_base_world, self.clientID)
        rot_body_world = transform_to_body_frame(rot_base_world)

        psi, theta, phi = compute_body_euler(rot_body_world)

        psi_dot, theta_dot, phi_dot = compute_body_velocity(rot_body_world, ang_vel_base_world)


        state = [lin_pos_base_world[0], lin_pos_base_world[1], psi-YAW_OFFSET, theta, phi,
                 lin_vel_base_world[0], lin_vel_base_world[1], psi_dot, theta_dot, phi_dot]

        #!!!!!!!!!MASS AND INERTIAL PARAMETERS SHOULD BE UPDATED HERE!!!!
        com_ke = 0.5*1.0*(lin_vel_base_world[0]**2+lin_vel_base_world[1]**2+lin_vel_base_world[2]**2)
        com_pe = 1.0*9.81*lin_pos_base_world[2]
        # rot_ke = compute_rotation_ke(ang_vel_base_world)
        self._z_COM = lin_pos_base_world[2]

        total_energy = com_ke + com_pe #+ rot_ke

        return state, total_energy

    def get_noisy_observation(self, np_random):

        cone_state, cone_te = self.get_observation()
        # mu = np.zeros([10,])
        return cone_state+np_random.normal(0.0,0.02,10) #0.05

    def get_rot_transform_s_b(self):
        cone_state, cone_te = self.get_observation()

        rot_psi = R.from_euler('z', cone_state[2]).as_matrix()
        rot_init = R.from_euler('z', np.pi/2).as_matrix()
        rot_theta = R.from_euler('y', cone_state[3]).as_matrix()

        rot_sb = np.matmul(np.matmul(rot_psi, rot_init),rot_theta)

        return rot_sb

    def generate_urdf_file(self):

        mass = 100

        cm_x = 0.059102
        cm_y = 0.131882
        cm_z = -1.663169

        ixx=1.298034
        ixy=0.012782
        ixz=0.006905
        iyy=1.418582
        iyz=0.008100
        izz=0.251987

        data = '''<?xml version="1.0"?>
<robot name="AutoObjectTransport">

  <material name="gray">
      <color rgba="0.66 0.66 0.66 1"/>
  </material>



  <link name="base_link">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0.0 0.0 0.0"/>
      <inertia ixx="1e-10" ixy="0." ixz="0." iyy="1e-10" iyz="0." izz="1e-10"/>
    </inertial>
  </link>

  <link name="apex_link_x">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0.0 0.0 0.0"/>
      <inertia ixx="1e-10" ixy="0." ixz="0." iyy="1e-10" iyz="0." izz="1e-10"/>
    </inertial>
  </link>

  <link name="apex_link_y">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0.0 0.0 0.0"/>
      <inertia ixx="1e-10" ixy="0." ixz="0." iyy="1e-10" iyz="0." izz="1e-10"/>
    </inertial>
  </link>

  <link name="apex_link_z">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0.0 0.0 0.0"/>
      <inertia ixx="1e-10" ixy="0." ixz="0." iyy="1e-10" iyz="0." izz="1e-10"/>
    </inertial>
  </link>

  <link name="apex_link_dummy_1">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0.0 0.0 0.0"/>
      <inertia ixx="1e-10" ixy="0." ixz="0." iyy="1e-10" iyz="0." izz="1e-10"/>
    </inertial>
  </link>

  <link name="apex_link_dummy_2">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0.0 0.0 0.0"/>
      <inertia ixx="1e-10" ixy="0." ixz="0." iyy="1e-10" iyz="0." izz="1e-10"/>
    </inertial>
  </link>

  <link name="cone">
    <inertial>
      <mass value="'''+str(mass)+'''"/>
      <origin xyz="'''+str(cm_x)+" "+str(cm_y)+" "+str(cm_z)+'''"/>
      <inertia ixx="'''+str(ixx)+'''" ixy="'''+str(ixy)+'''" ixz="'''+str(ixz)+'''" iyy="'''+str(iyy)+'''" iyz="'''+str(iyz)+'''" izz="'''+str(izz)+'''"/>
    </inertial>

    <contact>
      <lateral_friction value="0.4"/>
    </contact>

    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="mesh/moai_3m_tall/moai_bm_axis_change.obj"/>
      </geometry>
    </visual>

    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="mesh/moai_3m_tall/moai_bm_vhacd_axis_change.obj"/>
      </geometry>
    </collision>

  </link>


  <joint name="joint_apex_x" type="prismatic">
    <axis xyz="1 0 0"/>
    <parent link="base_link"/>
    <child link="apex_link_x"/>
    <limit lower="-100" upper="100"/>
  </joint>

  <joint name="joint_apex_y" type="prismatic">
    <axis xyz="0 1 0"/>
    <parent link="apex_link_x"/>
    <child link="apex_link_y"/>
    <limit lower="-100" upper="100"/>
  </joint>

  <joint name="joint_apex_z" type="prismatic">
    <axis xyz="0 0 1"/>
    <parent link="apex_link_y"/>
    <child link="apex_link_z"/>
    <limit lower="-100" upper="100"/>
  </joint>

  <joint name="joint_apex_dummy_1" type="spherical">
    <axis xyz="0 0 1"/>
    <parent link="apex_link_z"/>
    <child link="apex_link_dummy_1"/>
  </joint>

  <joint name="joint_apex_dummy_2" type="spherical">
    <axis xyz="0 1 0"/>
    <parent link="apex_link_dummy_1"/>
    <child link="apex_link_dummy_2"/>
  </joint>

  <joint name="joint_apex_dummy_3" type="spherical">
    <axis xyz="1 0 0"/>
    <parent link="apex_link_dummy_2"/>
    <child link="cone"/>
  </joint>

</robot>'''

        with open('./Rock-Walk/rock_walk/resources/models/auto_object.urdf', 'w') as f:
            f.write(data)




# def change_constraint(self):
#
#     lin_pos_base_world, quat_base_world = bullet.getBasePositionAndOrientation(self.objectID, self.clientID)
#     rot_base_world = bullet.getMatrixFromQuaternion(quat_base_world, self.clientID)
#
#     rot_base_world_np = np.array([[rot_base_world[0], rot_base_world[1], rot_base_world[2]],
#                                   [rot_base_world[3], rot_base_world[4], rot_base_world[5]],
#                                   [rot_base_world[6], rot_base_world[7], rot_base_world[8]]])
#
#     apex_vector = np.matmul(rot_base_world_np, np.array([0.0, -0.262339, 1.109297]))
#
#     apex_pos_base_world = [lin_pos_base_world[0]+apex_vector[0],
#                            lin_pos_base_world[1]+apex_vector[1],
#                            lin_pos_base_world[2]+apex_vector[2]]
#
#     bullet.changeConstraint(self.constraintID, apex_pos_base_world, quat_base_world)




# f_name1 = os.path.join(os.path.dirname(__file__),'models/large_cone_apex_control_a.urdf')
# f_name2 = os.path.join(os.path.dirname(__file__),'models/large_cone_apex_control_b.urdf')

#
# self.objectID = bullet.loadURDF(fileName=f_name1, basePosition=[0, 0, 1.50],
#                               baseOrientation=bullet.getQuaternionFromEuler([0,0,self._yaw_spawn]),#([0,0,np.pi/2]),
#                               physicsClientId=client)
#
# self.coneCtrlID = bullet.loadURDF(fileName=f_name2, basePosition=[0, 0, 1.50],
#                               baseOrientation=bullet.getQuaternionFromEuler([0,0,self._yaw_spawn]),#([0,0,np.pi/2]),
#                               useFixedBase=1,
#                               physicsClientId=client)
#
# self.constraintID = bullet.createConstraint(self.coneCtrlID, 3, self.objectID, -1, bullet.JOINT_POINT2POINT,
#                                             [0, 0, 0],[0, 0, 0], [0.0, -0.262339, 1.109297])
