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
        # dets: [x1, y1, x2, y2, conf]
        for t in self.tracks:
            t.predict()

        if len(dets) > 0:
            trks = np.array([t.history[-1][0] for t in self.tracks])
            matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(dets[:, :4], trks,
                                                                                     self.iou_threshold)

            for m in matched:
                self.tracks[m[1]].update(dets[m[0], :4], dets[m[0], 4])

            for i in unmatched_dets:
                self.tracks.append(Track(dets[i, :4], dets[i, 4]))

        ret = []
        i = len(self.tracks)
        for t in reversed(self.tracks):
            if t.time_since_update < 1 and (t.hits >= self.min_hits or t.age <= self.min_hits):
                b = t.history[-1][0]
                ret.append(np.concatenate(([t.id + 1], b, [t.conf])).reshape(1, -1))
            i -= 1
            if t.time_since_update > self.max_age:
                self.tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))