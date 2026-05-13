import numpy as np
from .kalman_filter import KalmanBoxTracker


class TrackState:
    Tentative = 1
    Confirmed = 2
    Lost = 3


class Track:
    count = 0

    def __init__(self, bbox, conf):
        self.id = Track.count
        Track.count += 1
        self.kf = KalmanBoxTracker(bbox)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.conf = conf
        self.state = TrackState.Tentative

    def predict(self):
        if (self.kf.kf.x[6] + self.kf.kf.x[2]) <= 0:
            self.kf.kf.x[6] *= 0.0
        self.kf.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.get_state()[0]

    def update(self, bbox, conf):
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf
        self.kf.update(bbox)

    def mark_missed(self, max_age):
        # Tentative tracks die immediately if not matched
        if self.state == TrackState.Tentative:
            self.state = TrackState.Lost
        # Confirmed tracks only die after max_age frames without update
        elif self.time_since_update > max_age:
            self.state = TrackState.Lost

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_lost(self):
        return self.state == TrackState.Lost

    def get_bbox(self):
        """Zwraca wygładzoną pozycję z filtru Kalmana [x1, y1, x2, y2]."""
        return self.kf.get_state()[0]