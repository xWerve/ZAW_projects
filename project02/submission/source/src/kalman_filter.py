import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    Kalman filter dla bounding boxa w przestrzeni [cx, cy, s, r, vx, vy, vs].
    Dotuningowany pod pieszych: wolniejsze obiekty, stabilniejsze predykcje.
    """

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise — rozmiar bbox bardziej zaszumiony niż pozycja
        self.kf.R[2:, 2:] *= 10.

        # Initial covariance — duża niepewność prędkości na starcie
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        # Process noise — piesi poruszają się płynnie, mała niepewność procesu
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.bbox_to_z(bbox)

        # Historia predykcji do obliczania smooth position
        self._last_bbox = bbox.copy()

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        predicted = self.z_to_bbox(self.kf.x)
        self._last_bbox = predicted[0]
        return predicted

    def update(self, bbox):
        self.kf.update(self.bbox_to_z(bbox))
        self._last_bbox = bbox.copy()

    def get_state(self):
        return self.z_to_bbox(self.kf.x)

    def bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))

    def z_to_bbox(self, x):
        w = np.sqrt(np.abs(x[2] * x[3]))
        h = x[2] / w if w > 0 else 0
        return np.array([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.
        ]).reshape((1, 4))