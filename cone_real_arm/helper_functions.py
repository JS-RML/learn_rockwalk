import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

def compute_tangent_plane_transform_at_C(obs, vec_GC_s):

    unit_vec_GC_s = vec_GC_s/np.linalg.norm(vec_GC_s)

    rot_vector = np.cross(np.array([0,0,1]), unit_vec_GC_s)
    unit_rot_vector = rot_vector/np.linalg.norm(rot_vector)

    rot_angle = np.arccos(np.dot(np.array([0,0,1]), unit_vec_GC_s))

    rot_mat = np.eye(3) + np.sin(rot_angle)*skew(unit_rot_vector) \
                        + (1-np.cos(rot_angle))*np.matmul(skew(unit_rot_vector), skew(unit_rot_vector))


    return np.matmul(rot_mat, R.from_euler('z', obs[0]+np.pi/2).as_matrix())



def get_vector_GC_s(obs, rot_sb):

    ellipse_params=[obs[6],obs[6]]
    apex_coordinates=[0,obs[7],obs[8]]


    rot_phi = R.from_euler('z', obs[2]).as_matrix()
    rot_sbprime = np.matmul(rot_sb, rot_phi)


    apex_coordinates_bprime = np.matmul(R.from_euler('z', np.pi/2).as_matrix(), np.array(apex_coordinates))


    ellipse_a = ellipse_params[0]
    ellipse_b = ellipse_params[1]
    num =  (ellipse_a*ellipse_b)
    denom = np.sqrt(np.square(ellipse_a*np.cos(obs[2]))+np.square(ellipse_b*np.sin(obs[2])))
    ellipse_r_phi =num/denom
    DG_b = np.array([ellipse_r_phi,0,0])


    ground_contact_s = np.matmul(rot_sb,DG_b)
    apex_coordinate_s = np.matmul(rot_sbprime, apex_coordinates_bprime)

    vec_GC = apex_coordinate_s-ground_contact_s

    return vec_GC


def get_rot_transform_s_b(obs):

    rot_psi = R.from_euler('z', obs[0]).as_matrix()
    rot_init = R.from_euler('z', np.pi/2).as_matrix()
    rot_theta = R.from_euler('y', obs[1]).as_matrix()

    rot_sb = np.matmul(np.matmul(rot_psi, rot_init),rot_theta)

    return rot_sb



def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
