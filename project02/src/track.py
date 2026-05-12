from .kalman_filter import KalmanBoxTracker

class Track:
    count = 0
    def __init__(self, bbox, conf):
        self.id = Track.count
        Track.count += 1
        self.kf = KalmanBoxTracker(bbox)
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.conf = conf
        self.history = []

    def predict(self):
        bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(bbox)
        return bbox[0]

    def update(self, bbox, conf):
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf
        self.kf.update(bbox)