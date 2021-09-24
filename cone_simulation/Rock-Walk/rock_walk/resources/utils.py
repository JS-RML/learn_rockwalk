import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_to_body_frame(rot_w):

    rot_w_np = np.array([[rot_w[0], rot_w[1], rot_w[2]],
                         [rot_w[3], rot_w[4], rot_w[5]],
                         [rot_w[6], rot_w[7], rot_w[8]]])

    z_rot = R.from_euler('z', -np.pi/2).as_matrix()

    return np.matmul(rot_w_np, z_rot)



def compute_body_euler(rot):
    """ Computes body eulers suitable to analyze rolling motion:
    Rot(z, \psi)*Rot(z,pi/2)*Rot(y, \theta)*Rot(z, \phi)
    Also take a look at the README.md file.
    """

    if rot[2,2]!= 1 or rot[2,2]!= 1:
        theta = math.atan2(math.sqrt(math.pow(rot[0,2],2) + math.pow(rot[1,2],2)),rot[2,2])
        phi = math.atan2(rot[2,1],-rot[2,0])
        psi = math.atan2(-rot[0,2],rot[1,2])

    elif rot[2,2]== -1:
        theta = math.pi/2
        phi = 0
        psi = math.atan2(rot[0,0], -rot[1,0])

    elif rot[2,2]== 1:
        theta = 0
        phi = 0
        psi = math.atan2(-rot[0,0],rot[1,0])

    return psi, theta, phi



def compute_body_velocity(rot, ang_vel):
    ang_vel_np = np.array([ang_vel[0], ang_vel[1], ang_vel[2]])
    ang_vel_body = np.dot(np.transpose(rot), ang_vel_np)

    psi_dot = ang_vel_body[0]
    theta_dot = ang_vel_body[1]
    phi_dot = ang_vel_body[2]

    return psi_dot, theta_dot, phi_dot


def compute_rotation_ke(ang_vel):
    ang_vel_np = np.array([ang_vel[0], ang_vel[1], ang_vel[2]])
    I = np.array([[0.21, 0., 0,],[0., 0.20, -0.05],[0., -0.05, 0.09]])

    rot_ke = np.dot(ang_vel_np, np.dot(I, ang_vel_np))

    return rot_ke
