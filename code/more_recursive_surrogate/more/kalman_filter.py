import numpy as np


class KalmanFilter:
    """Kalman Filter implementation based on
       "Bayesian Estimation of time-varying systems" by Simo Särkkä.
    """

    def __init__(self, m_0, P_0):

        # prior Distribution of state
        self.m_old = m_0
        self.prev_m = None
        self.P_old = P_0

        # prediction
        self.m_pred = m_0
        self.P_pred = P_0

    def momentum_prediction(self):
        # TODO: try it only once for each MORE iteration
        if self.prev_m is not None:
            diff = self.m_old - self.prev_m
            if all(abs(x) < 1 for x in diff):
                self.m_old = self.m_old + diff

    def predict(self, A, Q):
        """Performs prediction step of Kalman filter.

        Args:
            A: state transition model 
            Q: covariance of gaussian walk performed by parameters

        Returns:
            Mean and covariance of predicted posterior distribution (gaussian).
        """
        # TODO: try momentum based (difference) prediction step
        # self.momentum_prediction()

        self.m_pred = A @ self.m_old
        self.P_pred = A @ self.P_old @ A.T + Q
        return self.m_pred, self.P_pred

        
    def update(self, H, R, y):
        """Performs update step of Kalman filter.
        
        Args:
            H: measurements matrix
            R: covariance matrix for measurements
            y: targets
        Returns:
            Mean and covariance of updated posterior distribution (gaussian)
        """
        # temporary variables
        S = H @  self.P_pred @ H.T + R
        if np.isscalar(S):
            K = self.P_pred @ H.T * (S**-1)
        else:
            K = self.P_pred @ H.T @ np.linalg.inv(S)
        
        # mean of new posterior distribution of the parameters
        if np.isscalar(y) or len(y) == 1:
            m = self.m_pred + K * (y - H @ self.m_pred)
        else:
            m = self.m_pred + K @ (y - H @ self.m_pred)
        self.prev_m = self.m_old
        self.m_old = m

        # covariance of new posterior distribution of the parameters
        if np.isscalar(S):
            P = self.P_pred - K @ (S * K.T)
        else:
            P = self.P_pred - K @ S @ K.T
        self.P_old = P

        return m, P

