# Robust Pose Estimation from a Planar Target (RPP)
# Based on the work of G Schweighofer & A Pinz (2006)
# Ported to Python from the MATLAB code by M Holmes

import numpy as np
from rpputil import *


# Main RPP function
# Input:    (3 x n) 3D object points, (3 x n) 2D image points, initial (3 x 3) guess of the rotation matrix,
#           float64 tolerance for recursion, float64 epsilon for recursion,
#           'SVD' (single value decomposition) or 'QRN' (quaternion) for rotation solve method
# Output:   Numpy Tuple (R, t, err) where R = 3x3 Rotation matrix, t = 3x1 translation vector, err = projection error
def rpp(objPoints=None, imgPoints=None, initRGuess=None, tolerance=None, epsilon=None, rotMethod=None, maxIter=None):
    # Use test data if no values given
    if objPoints is None:
        objPoints = np.array([0.0685, 0.6383, 0.4558, 0.7411, -0.7219, 0.7081, 0.7061, 0.2887, -0.9521, -0.2553,
                              0.4636, 0.0159, -0.1010, 0.2817, 0.6638, 0.1582, 0.3925, -0.7954, 0.6965, -0.7795],
                             dtype=np.float64).reshape((2, 10))
        objPoints = np.vstack((objPoints, np.zeros((1, objPoints.shape[1]))))

    if imgPoints is None:
        imgPoints = np.array([-0.0168, 0.0377, 0.0277, 0.0373, -0.0824, 0.0386, 0.0317, 0.0360, -0.1015, -0.0080,
                              0.0866, 0.1179, 0.1233, 0.1035, 0.0667, 0.1102, 0.0969, 0.1660, 0.0622, 0.1608],
                             dtype=np.float64).reshape((2, 10))

    # Use defaults if no tolerance/epsilon/rotation method set set
    if tolerance is None:
        tolerance = 1e-5
    if epsilon is None:
        epsilon = 1e-8
    if rotMethod is None:
        rotMethod = 'SVD'
    if maxIter is None:
        maxIter = 40

    options = {'method': rotMethod, 'tolerance': tolerance, 'epsilon': epsilon, 'maxIter': maxIter}
    if initRGuess is not None:
        options['initRGuess'] = initRGuess

    # Get a first guess of the pose
    rotMat, transMat, numIter, objErr, imgErr = obj_pose(objPoints.copy(), imgPoints.copy(), options)

    # Get second pose
    imgPointsAug = np.vstack((imgPoints, np.ones((1, imgPoints.shape[1]))))
    sol = get_2nd_pose_exact(imgPointsAug, objPoints.copy(), rotMat, transMat, 0)

    # Refine poses
    for idx in range(len(sol)):
        options['initRGuess'] = sol[idx]['R']
        Rlu, tlu, it1_, objErr1, imgErr1 = obj_pose(objPoints.copy(), imgPoints, options)
        sol[idx]['R'] = Rlu
        sol[idx]['t'] = tlu
        sol[idx]['objErr'] = objErr1

    e = []
    for idx in range(len(sol)):
        e.append((sol[idx]['objErr'], idx))
    e = np.array(e, dtype=[('error', 'f8'), ('idx', 'i4')])
    e = np.sort(e, order='error')
    bestSol = sol[e[0]['idx']]

    return bestSol['R'], bestSol['t'], bestSol['objErr']


# Object Pose Estimation Function
# Implements the algorithm described in "Fast and Globally Convergent Pose Estimation from Video Images" by Chien-Ping
# Lu et. al. to appear in IEEE Transaction on Pattern Analysis and Machine Intelligence
#
# Inputs: (3 x n) 3D object points, (2 x n) 2D image points, options dictionary
# Options: 'initRGuess': Initial 3x3 rotation matrix estimate,
#          'method': 'SVD' to use Single Value Decomposition for solving rotation, 'QTN' to use Quaternions
#          'tolerance': Convergence tolerance, by default 1e-5
#          'epsilon': Convergence tolerance, by default 1e-8
def obj_pose(objPoints, imgPoints, options):
    # Get number of points
    n = objPoints.shape[1]

    # Move object point origin to center of object points
    pbar = (np.sum(objPoints, 1) / n).reshape((3, 1))
    for idx in range(n):
        objPoints[:, idx] -= pbar[:, 0]
    Q = np.vstack((imgPoints, np.ones((1, n))))

    # Compute projection matrices
    F = np.zeros((3, 3, n))
    for idx in range(n):
        V = np.array(Q[:, idx] / Q[2, idx]).reshape((3, 1))
        F[:, :, idx] = np.dot(V, V.T) / np.dot(V.T, V)

    # Compute the matrix factor required to compute translation vector t
    tFactor = np.linalg.inv(np.eye(3) - (np.sum(F, 2) / n)) / n
    it = 0
    if 'initRGuess' in options:
        # Use given initial rotation matrix guess
        Ri = options['initRGuess']
        sum_ = np.zeros((3, 1))
        for idx in range(n):
            sum_ += np.dot(F[:, :, idx] - np.eye(3), np.dot(Ri, objPoints[:, idx])).reshape((3, 1))
        ti = np.dot(tFactor, sum_)

        # Calculate error
        Qi = x_form(objPoints, Ri, ti)
        oldErr = 0
        for idx in range(n):
            vec = np.dot(np.eye(3) - F[:, :, idx], Qi[:, idx])
            oldErr += np.dot(vec, vec)
    else:
        # No initial guess; use weak-perspective approximation
        # Compute initial pose estimate
        Ri, ti, Qi, oldErr = abs_kernel(objPoints, Q, F, tFactor, options['method'])
        it = 1

    # Compute next post estimate
    Ri, ti, Qi, newErr = abs_kernel(objPoints, Qi, F, tFactor, options['method'])
    it += 1

    while (np.abs((oldErr - newErr) / oldErr) > options['tolerance']) and (newErr > options['epsilon']) and \
                    it < options['maxIter']:
        oldErr = newErr
        # Compute the optimal estimate of R
        Ri, ti, Qi, newErr = abs_kernel(objPoints, Qi, F, tFactor, options['method'])
        it += 1

    R = Ri
    t = ti
    objErr = np.sqrt(newErr / n)

    Qproj = x_form_proj(objPoints, R, t)
    imgErr = 0
    QprojFlat = Qproj.flatten(1)
    imgPointsFlat = imgPoints.flatten(1)
    for idx in range(n):
        vec = QprojFlat[idx] - imgPointsFlat[idx]
        imgErr += np.dot(vec, vec)
    imgErr = np.sqrt(imgErr / n)

    # Get back to the original reference frame
    t -= np.dot(Ri, pbar).reshape((3, 1))

    return R, t, it, objErr, imgErr


def estimate_t(R, G, F, P, n):
    sum_ = np.zeros((3, 1))
    for idx in range(n):
        sum_ += np.dot(F[:, :, idx], np.dot(R, P[:, idx])).reshape((3, 1))
    return np.dot(G, sum_)


def abs_kernel(P, Q, F, G, method):
    n = P.shape[1]

    for idx in range(n):
        Q[:, idx] = np.dot(F[:, :, idx], Q[:, idx])
    # Compute P' and Q'
    pbar = np.sum(P, 1) / n
    test = np.sum(P[0, :])
    qbar = np.sum(Q, 1) / n
    for idx in range(n):
        P[:, idx] = P[:, idx] - pbar
        Q[:, idx] = Q[:, idx] - qbar

    if method == 'SVD':  # Use Single Value Decomposition solution
        M = np.zeros((3, 3))
        for idx in range(n):
            M += np.dot(P[:, idx].reshape((3, 1)), Q[:, idx].reshape((3, 1)).T)

        # Calculate single value decomposition of M
        U, S, V = np.linalg.svd(M)
        V = V.T

        # Compute rotation matrix R
        R = np.dot(V, U.T)
        if np.linalg.det(R) > 0:
            t = estimate_t(R, G, F, P, n)
            if t[2] < 0:
                R = np.dot(-np.hstack((V[:, :2], -V[:, 2].reshape((3, 1)))), U.T)
                t = estimate_t(R, G, F, P, n)
        else:
            R = np.dot(np.hstack((V[:, :2], -V[:, 2].reshape((3, 1)))), U.T)
            t = estimate_t(R, G, F, P, n)
            if t[2] < 0:
                R = -np.dot(V, U.T)
                t = estimate_t(R, G, F, P, n)

    elif method == 'QTN':
        # Compute M Matrix
        A = np.zeros((4, 4))
        for idx in range(n):
            A += np.dot(q_mat_Q(np.vstack((np.array([1]), Q[:, idx]))).T,
                        q_mat_W(np.vstack((np.array([1]), P[:, idx]))))

        # Find the largest Eigenvalue of A
        V, D = np.linalg.eig(A)
        # TODO: Make sure largest eigenvalue

        # Compute rotation matrix R from the quaternion that corresponds to the largest eigenvalue of A
        R = quat_2_mat(V)
        sum_ = np.zeros((3, 1))
        for idx in range(n):
            sum_ += np.dot(F[:, :, idx], np.dot(R, P[:, idx]))
        t = np.dot(G, sum_)

    Qout = x_form(P, R, t)
    err2 = 0
    for idx in range(n):
        vec = np.dot(np.eye(3) - F[:, :, idx], Qout[:, idx])
        err2 += np.dot(vec, vec)

    return R, t, Qout, err2


# Returns second pose if a first pose was calculated
def get_2nd_pose_exact(v, P, R, t, DB):
    cent = norm_Rv(np.mean(norm_Rv(v), 1).reshape((3, 1)))
    Rim = get_rotation_by_vector(np.array([0, 0, 1]).reshape((3, 1)), cent)

    v_ = np.dot(Rim, v)
    cent = norm_Rv(np.mean(norm_Rv(v_), 1).reshape((3, 1)))

    R_ = np.dot(Rim, R)
    t_ = np.dot(Rim, t)

    sol = get_R_for_2nd_pose_V_exact(v_, P, R_, t_, DB)

    # De-normalise the pose
    for idx in range(len(sol)):
        sol[idx]['R'] = np.dot(Rim.conj().T, sol[idx]['R'])
        sol[idx]['t'] = np.dot(Rim.conj().T, sol[idx]['t'])

    return sol


# Gets the exact R with variations in t
def get_R_for_2nd_pose_V_exact(v, P, R, t, DB):
    RzN = decompose_R(R)
    R_ = np.dot(R, RzN)

    # Change model by Rz
    P_ = np.dot(RzN.conj().T, P)

    # Project into image with only Ry
    angZYX = rpy_ang_X(R_)
    Ry = rpy_mat(np.array([0, angZYX[1], 0]).reshape((3, 1)))
    Rz = rpy_mat(np.array([0, 0, angZYX[2]]).reshape((3, 1)))

    bl, TNew, at = get_rotation_Y_wrt_T(v, P_, t, DB, Rz)

    # We got 2 solutions. YEAH
    V = np.zeros((3, 3, v.shape[1]))
    for idx in range(v.shape[1]):
        tmp = v[:, idx].reshape((3, 1))
        a = np.dot(tmp.T, tmp)
        V[:, :, idx] = np.dot(tmp, tmp.T) / a[0]

    sol = []
    for idx2 in range(len(bl)):
        sol.append({})
        sol[idx2]['bl'] = bl[idx2]
        sol[idx2]['at'] = at[idx2]

        Ry = rpy_mat(np.array([0, bl[idx2], 0]).reshape((3, 1)))
        sol[idx2]['R'] = np.dot(Rz, np.dot(Ry, RzN.conj().T))
        sol[idx2]['t'] = TNew[:, idx2]

        # Test the error
        E = 0
        for idx1 in range(v.shape[1]):
            E += np.sum(np.dot(np.eye(3) - V[:, :, idx1], np.dot(sol[idx2]['R'], P[:, idx1]) + sol[idx2]['t']) ** 2)
        sol[idx2]['E'] = E
    return sol


def decompose_R(R):
    cl = np.arctan2(R[2, 1], R[2, 0])
    Rz = rpy_mat(np.array([0, 0, cl]).reshape((3, 1)))
    return Rz


# returns a minimization of e = sum( (I-Vi)*(Ry*Pi+t)).^2
def get_rotation_Y_wrt_T(v, p, t, DB, Rz=None):
    if Rz is None:
        Rz = np.eye(3)

    # generate Vi
    V = np.zeros((3, 3, v.shape[1]))
    for idx in range(v.shape[1]):
        vIdx = v[:, idx].reshape((3, 1))
        V[:, :, idx] = np.dot(vIdx, vIdx.conj().T) / np.dot(vIdx.conj().T, vIdx)

    test = V[:, :, 0]
    # generate G
    G = np.zeros((3, 3))
    for idx in range(v.shape[1]):
        G += V[:, :, idx]
    G = np.linalg.inv(np.eye(3) - G / v.shape[1]) / v.shape[1]

    # generate opt_t*[bt^2 bt 1]
    opt_t = np.zeros((3, 3))
    for idx in range(v.shape[1]):
        v11 = V[0, 0, idx]
        v12 = V[0, 1, idx]
        v13 = V[0, 2, idx]
        v21 = V[1, 0, idx]
        v22 = V[1, 1, idx]
        v23 = V[1, 2, idx]
        v31 = V[2, 0, idx]
        v32 = V[2, 1, idx]
        v33 = V[2, 2, idx]

        px = p[0, idx]
        py = p[1, idx]
        pz = p[2, idx]

        # generate opt_t*[bt^2 bt 1] with new Rz value
        if True:
            r1 = Rz[0, 0]
            r2 = Rz[0, 1]
            r3 = Rz[0, 2]
            r4 = Rz[1, 0]
            r5 = Rz[1, 1]
            r6 = Rz[1, 2]
            r7 = Rz[2, 0]
            r8 = Rz[2, 1]
            r9 = Rz[2, 2]

            opt_t += np.array([(((v11 - 1) * r2 + v12 * r5 + v13 * r8) * py + (
            -(v11 - 1) * r1 - v12 * r4 - v13 * r7) * px + (-(v11 - 1) * r3 - v12 * r6 - v13 * r9) * pz),
                               ((2 * (v11 - 1) * r1 + 2 * v12 * r4 + 2 * v13 * r7) * pz + (
                               -2 * (v11 - 1) * r3 - 2 * v12 * r6 - 2 * v13 * r9) * px),
                               ((v11 - 1) * r1 + v12 * r4 + v13 * r7) * px + (
                               (v11 - 1) * r3 + v12 * r6 + v13 * r9) * pz + ((v11 - 1) * r2 + v12 * r5 + v13 * r8) * py,
                               ((v21 * r2 + (v22 - 1) * r5 + v23 * r8) * py + (
                               -v21 * r1 - (v22 - 1) * r4 - v23 * r7) * px + (
                                -v21 * r3 - (v22 - 1) * r6 - v23 * r9) * pz),
                               ((2 * v21 * r1 + 2 * (v22 - 1) * r4 + 2 * v23 * r7) * pz + (
                               -2 * v21 * r3 - 2 * (v22 - 1) * r6 - 2 * v23 * r9) * px),
                               (v21 * r1 + (v22 - 1) * r4 + v23 * r7) * px + (
                               v21 * r3 + (v22 - 1) * r6 + v23 * r9) * pz + (v21 * r2 + (v22 - 1) * r5 + v23 * r8) * py,
                               ((v31 * r2 + v32 * r5 + (v33 - 1) * r8) * py + (
                               -v31 * r1 - v32 * r4 - (v33 - 1) * r7) * px + (
                                -v31 * r3 - v32 * r6 - (v33 - 1) * r9) * pz),
                               ((2 * v31 * r1 + 2 * v32 * r4 + 2 * (v33 - 1) * r7) * pz + (
                               -2 * v31 * r3 - 2 * v32 * r6 - 2 * (v33 - 1) * r9) * px),
                               (v31 * r1 + v32 * r4 + (v33 - 1) * r7) * px + (
                               v31 * r3 + v32 * r6 + (v33 - 1) * r9) * pz + (
                               v31 * r2 + v32 * r5 + (v33 - 1) * r8) * py]).reshape((3, 3))
        else:
            opt_t += np.array([(v12 * py + (1 - v11) * px - v13 * pz), ((2 * v11 - 2) * pz - 2 * v13 * px),
                               (v11 - 1) * px + v13 * pz + v12 * py,
                               ((v22 - 1) * py - v21 * px - v23 * pz), (2 * v21 * pz - 2 * v23 * px),
                               v21 * px + v23 * pz + (v22 - 1) * py,
                               (v32 * py - v31 * px + (1 - v33) * pz), (2 * v31 * pz + (-2 * v33 + 2) * px),
                               v31 * px + (v33 - 1) * pz + v32 * py]).reshape((3, 3))

    opt_t = np.dot(G, opt_t)

    E_2 = np.zeros((1, 5))
    # estimate Error function E
    for idx in range(v.shape[1]):
        v11 = V[0, 0, idx]
        v12 = V[0, 1, idx]
        v13 = V[0, 2, idx]
        v21 = V[1, 0, idx]
        v22 = V[1, 1, idx]
        v23 = V[1, 2, idx]
        v31 = V[2, 0, idx]
        v32 = V[2, 1, idx]
        v33 = V[2, 2, idx]

        px = p[0, idx]
        py = p[1, idx]
        pz = p[2, idx]

        # R*pi;
        Rpi = np.array([-px, 2 * pz, px,
                        py, 0, py,
                        -pz, -2 * px, pz]).reshape((3, 3))

        E = np.dot((np.eye(3) - V[:, :, idx]), np.dot(Rz, Rpi) + opt_t)

        # get E.^2
        e2 = E[:, 0].reshape((3, 1))
        e1 = E[:, 1].reshape((3, 1))
        e0 = E[:, 2].reshape((3, 1))

        E_2 += np.sum(np.hstack((e2 ** 2, 2 * e1 * e2, (2 * e0 * e2 + e1 ** 2), 2 * e0 * e1, e0 ** 2)), 0)

    E_2 = E_2.reshape(5)
    e4 = E_2[0]
    e3 = E_2[1]
    e2 = E_2[2]
    e1 = E_2[3]
    e0 = E_2[4]

    a4 = -e3
    a3 = (4 * e4 - 2 * e2)
    a2 = (-3 * e1 + 3 * e3)
    a1 = (-4 * e0 + 2 * e2)
    a0 = e1

    # Solve for roots of polynomial equation
    roots = np.roots([a4, a3, a2, a1, a0]).reshape((4, 1))

    # get all valid solutions -> which are real zero
    e = a4 * roots ** 4 + a3 * roots ** 3 + a2 * roots ** 2 + a1 * roots + a0
    roots = roots[np.nonzero(np.abs(e) < 1e-3)[0]]

    # check if we have valid solutions
    p1 = (1 + roots ** 2) ** 3
    roots = roots[np.nonzero(np.abs(np.real(p1)) > 0.1)[0]]

    sa = np.real((2 * roots) / (1 + roots ** 2))
    ca = np.real((1 - roots ** 2) / (1 + roots ** 2))

    al = np.arctan2(sa, ca)

    tMaxMin = np.real(4 * a4 * roots ** 3 + 3 * a3 * roots ** 2 + 2 * a2 * roots + a1)

    al = al[np.nonzero(tMaxMin > 0)[0]]
    roots = roots[np.nonzero(tMaxMin > 0)[0]]

    tNew = np.zeros((3, len(al)))
    for a in range(len(al)):
        R = np.dot(Rz, rpy_mat(np.vstack((0, al[a], 0))))
        t_opt = np.zeros((3, 1))
        for i in range(v.shape[1]):
            t_opt += np.dot(V[:, :, i] - np.eye(3), np.dot(R, p[:, i]).reshape((3, 1)))
        t_opt = np.dot(G, t_opt).reshape((3, 1))
        tNew[:, a] = t_opt[:, 0]

    return al, tNew, roots


if __name__ == '__main__':
    sol = rpp()
