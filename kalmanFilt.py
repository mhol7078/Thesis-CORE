import numpy as np

__author__ = 'Michael Holmes'


class KalmanTrack:
    def __init__(self, initState, initControl, initP, posSTD, velSTD, accelSTD, rangeSTD, predictStep, updateStep):
        self.x = initState  # State vector
        self.u = initControl  # Control Vector
        self.P = initP  # State uncertainty matrix
        self.Q = np.diagflat(
            np.array([posSTD, posSTD, velSTD, velSTD, accelSTD, accelSTD]) ** 2)  # Prediction uncertainty matrix
        self.R = np.diagflat(np.array([rangeSTD, rangeSTD]) ** 2)  # Observation uncertainty matrix
        self.predictStep = predictStep  # Update rate in seconds for prediction stage
        self.updateStep = updateStep  # Update rate in seconds for update stage
        self.timeSincePredict = 0  # Time since last predict run in seconds
        self.timeSinceUpdate = 0  # Time since last update run in seconds
        self.maxUV = (0, 0)  # Max bounds on pixel position

    def predict(self):
        # Update state prediction
        xPredicted = self.predict_model()
        # Compute update for state uncertainty
        predJacobian = self.jacobian_predict()
        self.P = np.dot(np.dot(predJacobian, self.P), predJacobian.T) + self.Q
        self.x = self.constrain_position(xPredicted)
        # Reset predict elapsed counter
        self.timeSincePredict = 0
        return self.x, self.P

    def update(self, z):
        # Update filter
        zHat = self.observe_model()
        innov = z - zHat
        observJacobian = self.jacobian_observe()
        xUpdate, pUpdate = self.cholesky_update(innov, observJacobian)
        self.x = self.constrain_position(xUpdate)
        self.P = pUpdate
        # Update update elapsed counter
        self.timeSinceUpdate = 0
        return self.x, self.P

    def predict_model(self):
        # Constant Acceleration Model is as generic as possible for model-less dynamics
        stateOut = np.zeros((6, 1))
        stateOut[0] = self.x[0] + self.x[2] * self.timeSincePredict + 0.5 * self.x[4] * (self.timeSincePredict ** 2)
        stateOut[1] = self.x[1] + self.x[3] * self.timeSincePredict + 0.5 * self.x[5] * (self.timeSincePredict ** 2)
        stateOut[2] = self.x[2] + self.x[4] * self.timeSincePredict
        stateOut[3] = self.x[3] + self.x[5] * self.timeSincePredict
        stateOut[4] = self.x[4]
        stateOut[5] = self.x[5]
        return stateOut

    def jacobian_predict(self):
        # Analytical time derivative of constant acceleration model (fortunately linear)
        jacobianOut = np.diagflat(np.ones((1, 6))) + np.diagflat(self.timeSincePredict * np.ones((1, 4)),
                                                                 2) + np.diagflat(
            0.5 * (self.timeSincePredict ** 2) * np.ones((1, 2)), 4)
        return jacobianOut

    def observe_model(self):
        return self.x[:2]

    def jacobian_observe(self):
        jacobianOut = np.zeros((2, 6))
        jacobianOut[0, 0] = 1
        jacobianOut[1, 1] = 1
        return jacobianOut

    def cholesky_update(self, v, H):
        # Adapted from Matlab code by Tim Bailey (2003)
        PH = np.dot(self.P, H.T)
        S = np.dot(H, PH) + self.R
        SChol = np.linalg.cholesky(S).T
        SCholInv = np.linalg.inv(SChol)
        WChol = np.dot(PH, SCholInv)
        W = np.dot(WChol, SCholInv.T)
        xUpdate = self.x + np.dot(W, v)
        pUpdate = self.P - np.dot(WChol, WChol.T)
        return xUpdate, pUpdate

    def constrain_position(self, state):
        if state[0] < 0:
            state[0] = 0
        if state[0] > self.maxUV[0]:
            state[0] = self.maxUV[0]
        if state[1] < 0:
            state[1] = 0
        if state[1] > self.maxUV[1]:
            state[1] = self.maxUV[1]
        return state

    def update_elapsed_counters(self, deltaTime):
        self.timeSincePredict += deltaTime
        self.timeSinceUpdate += deltaTime
        return

    def predict_stage_elapsed(self):
        if self.timeSincePredict >= self.predictStep:
            return True
        else:
            return False

    def update_stage_elapsed(self):
        if self.timeSinceUpdate >= self.updateStep:
            return True
        else:
            return False
