import numpy as np
from .track import Track
from .association import associate_detections_to_tracks


class Tracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, dets):
        for t in self.tracks:
            t.predict()

        if len(dets) > 0:
            trks = np.array([t.history[-1][0] for t in self.tracks]) if len(self.tracks) > 0 else np.empty((0, 4))
            matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(dets[:, :4], trks,
                                                                                     self.iou_threshold)

            for m in matched:
                self.tracks[m[1]].update(dets[m[0], :4], dets[m[0], 4])

            for i in unmatched_dets:
                self.tracks.append(Track(dets[i, :4], dets[i, 4]))

        ret = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]

            if t.time_since_update < 1 and (t.hits >= self.min_hits or t.age <= self.min_hits):
                b = t.history[-1][0]  # [x1, y1, x2, y2]
                ret.append(np.array([t.id + 1, b[0], b[1], b[2], b[3], t.conf]))

            if t.time_since_update > self.max_age:
                self.tracks.pop(i)

        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 6))