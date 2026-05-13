import numpy as np
from .track import Track, TrackState
from .association import associate_detections_to_tracks


class Tracker:
    def __init__(self,
                 max_age=40,
                 min_hits=1,
                 iou_threshold=0.3,
                 high_det_thresh=0.4,
                 low_det_thresh=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_det_thresh = high_det_thresh
        self.low_det_thresh = low_det_thresh
        self.tracks = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1

        for t in self.tracks:
            t.predict()

        if len(dets) == 0:
            for t in self.tracks:
                t.mark_missed(self.max_age)
            self._cleanup_tracks()
            return self._get_active_tracks()

        high_mask = dets[:, 4] >= self.high_det_thresh
        low_mask = (dets[:, 4] >= self.low_det_thresh) & ~high_mask
        dets_high = dets[high_mask]
        dets_low = dets[low_mask]

        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        other_tracks     = [t for t in self.tracks if not t.is_confirmed()]

        updated = set()

        unmatched_dets_high = list(range(len(dets_high)))
        unmatched_conf = list(range(len(confirmed_tracks)))

        if len(confirmed_tracks) > 0 and len(dets_high) > 0:
            trk_bboxes = np.array([t.get_bbox() for t in confirmed_tracks])
            age_penalty = np.array([
                min(t.time_since_update * 0.05, 0.3) for t in confirmed_tracks
            ])
            matched1, um_dh, um_conf = associate_detections_to_tracks(
                dets_high[:, :4], trk_bboxes,
                iou_threshold=self.iou_threshold,
                age_penalty=age_penalty
            )
            unmatched_dets_high = list(um_dh)
            unmatched_conf = list(um_conf)

            for m in matched1:
                t = confirmed_tracks[m[1]]
                t.update(dets_high[m[0], :4], dets_high[m[0], 4])
                updated.add(id(t))

        stage2_tracks = [confirmed_tracks[i] for i in unmatched_conf] + other_tracks

        if len(dets_low) > 0 and len(stage2_tracks) > 0:
            s2_bboxes = np.array([t.get_bbox() for t in stage2_tracks])
            matched2, _, _ = associate_detections_to_tracks(
                dets_low[:, :4], s2_bboxes,
                iou_threshold=max(0.1, self.iou_threshold - 0.1),
                use_giou=True
            )
            for m in matched2:
                t = stage2_tracks[m[1]]
                t.update(dets_low[m[0], :4], dets_low[m[0], 4])
                if t.hits >= self.min_hits:
                    t.state = TrackState.Confirmed
                updated.add(id(t))

        for t in self.tracks:
            if id(t) not in updated:
                t.mark_missed(self.max_age)

        for i in unmatched_dets_high:
            new_track = Track(dets_high[i, :4], dets_high[i, 4])
            if self.min_hits <= 1:
                new_track.state = TrackState.Confirmed
            self.tracks.append(new_track)

        self._cleanup_tracks()
        return self._get_active_tracks()

    def _cleanup_tracks(self):
        self.tracks = [t for t in self.tracks if not t.is_lost()]

    def _get_active_tracks(self):
        ret = []
        for t in self.tracks:
            if t.is_confirmed():
                b = t.get_bbox()
                ret.append(np.array([t.id + 1, b[0], b[1], b[2], b[3], t.conf]))
        if ret:
            return np.stack(ret)
        return np.empty((0, 6))