# Utility functions for the RPP code
import numpy as np


# Takes a 3x1 roll, pitch, yaw vector and converts it to 3x3 rotation matrix
def rpy_mat(angles):
    cosA = np.cos(angles[2])
    sinA = np.sin(angles[2])
    cosB = np.cos(angles[1])
    sinB = np.sin(angles[1])
    cosC = np.cos(angles[0])
    sinC = np.sin(angles[0])
    cosAsinB = cosA * sinB
    sinAsinB = sinA * sinB
    return np.array([cosA * cosB, cosAsinB * sinC - sinA * cosC, cosAsinB * cosC + sinA * sinC,
                     sinA * cosB, sinAsinB * sinC + cosA * cosC, sinAsinB * cosC - cosA * sinC,
                     -sinB, cosB * sinC, cosB * cosC], dtype=np.float32).reshape((3, 3))


# Returns a set of Roll, Pitch and Yaw angles that describe a certain 3x3
# transformation matrix. The magnitude of the Pitch angle is constrained
# to be not bigger than pi/2.
def rpy_ang(R):
    sinB = -R[2, 0]
    cosB = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if np.abs(cosB) > 1e-15:
        sinA = R[1, 0] / cosB
        cosA = R[0, 0] / cosB
        sinC = R[2, 1] / cosB
        cosC = R[2, 2] / cosB
        angles = np.array([np.arctan2(sinC, cosC), np.arctan2(sinB, cosB), np.arctan2(sinA, cosA)]).reshape((3, 1))
    else:
        sinC = (R[0, 1] - R[1, 2]) / 2
        cosC = (R[1, 1] + R[0, 2]) / 2
        angles = np.array([np.arctan2(sinC, cosC), np.pi / 2, 0]).reshape((3, 1))
        if sinB < 0:
            angles = -angles
    return angles


# Same as rpy_ang(R);  But: minimizes Rx(al)
def rpy_ang_X(R):
    angZYX = rpy_ang(R)
    if np.abs(angZYX[0]) > np.pi / 2:
        # test the same R
        while np.abs(angZYX[0]) > np.pi / 2:
            if angZYX[0] > 0:
                angZYX[0] = angZYX[0] + np.pi
                angZYX[1] = 3 * np.pi - angZYX[1]
                angZYX[2] = angZYX[2] + np.pi
                angZYX = angZYX - 2 * np.pi * np.ones_like(angZYX)
            else:
                angZYX[0] = angZYX[0] + np.pi
                angZYX[1] = 3 * np.pi - angZYX[1]
                angZYX[2] = angZYX[2] + np.pi
    return angZYX


# Compute the Q matrix (4 x 4) of quaternion q
def q_mat_Q(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return np.array([w, -x, -y, -z,
                     x, w, -z, y,
                     y, z, w, -x,
                     z, -y, x, w]).reshape((4, 4))


# Compute the W matrix (4 x 4) of quaternion q
def q_mat_W(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return np.array([w, -x, -y, -z,
                     x, w, z, -y,
                     y, -z, w, x,
                     z, y, -x, w]).reshape((4, 4))


# Convert a quaternion to a 3x3 rotation matrix
def quat_2_mat(q):
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    return np.array([a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (b * d + a * c),
                     2 * (b * c + a * d), a ** 2 + c ** 2 - b ** 2 - d ** 2, 2 * (c * d - a * b),
                     2 * (b * d - a * c), 2 * (c * d + a * b), a ** 2 + d ** 2 - b ** 2 - c ** 2]).reshape((3, 3))


# Transform the 3D point set P by rotation R and translation t
def x_form(P, R, t):
    n = P.shape[1]
    Q = np.zeros((3, n))
    for idx in range(n):
        Q[:, idx] = (np.dot(R, P[:, idx]).reshape((3, 1)) + t)[:, 0]
    return Q


# Transform the 3D point set P by rotation R and translation t, then project them to the normalized image plane
def x_form_proj(P, R, t):
    n = P.shape[1]
    Q = np.zeros((3, n))
    Qp = np.zeros((2, n))
    for idx in range(n):
        Q[:, idx] = (np.dot(R, P[:, idx].reshape((3, 1))) + t)[:, 0]
        Qp[:, idx] = Q[:2, idx] / Q[2, idx]
    return Qp


# Input (2 x n) Array of imagepoints
def norm_Rv(V):
    n = V.shape[1]
    v = np.zeros_like(V)
    for idx in range(n):
        mag = np.sum(V[:, idx] ** 2)
        mag = 1 / np.sqrt(mag)
        v[:, idx] = V[:, idx] * mag
    return v


# returns R so that v1 = R * v2;
def get_rotation_by_vector(v1, v2):
    v1 = v1.reshape(3)
    v2 = v2.reshape(3)
    winkel = np.arccos(v2.dot(v1))
    QU = quaternion_by_angle_and_vector(winkel, np.cross(v2, v1))
    R = quat_2_mat(np.vstack((QU['scalar'], QU['vector'].reshape((3, 1)), QU['scalar'])))
    return R


# Construct a normalized quaternion that rotates by an angle of qAngle around the axis qVector.
def quaternion_by_angle_and_vector(qAngle, qVector):
    rotationAxis = qVector / np.linalg.norm(qVector)
    Q_ = quaternion_by_vector_and_scalar(rotationAxis * np.sin(qAngle / 2), np.cos(qAngle / 2))
    Q = quaternion_multiply_by_scalar(Q_, 1 / quaternion_norm(Q_))
    return Q


# Combine vector and scalar into quaternion dictionary
def quaternion_by_vector_and_scalar(vector, scalar):
    q = {'vector': vector, 'scalar': scalar}
    return q


# Multiply a quaternion by a given scalar
def quaternion_multiply_by_scalar(q, scalar):
    Q = quaternion_by_vector_and_scalar(scalar * q['vector'], scalar * q['scalar'])
    return Q


# Compute the L2 norm of the given quaternion
def quaternion_norm(Q):
    nm = np.sqrt(np.linalg.norm(Q['vector']) ** 2 + Q['scalar'] ** 2)
    return nm
