import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes1, bboxes2):
    """Wektorowe IoU między dwoma zbiorami bboxów [x1,y1,x2,y2]."""
    bboxes1 = np.expand_dims(bboxes1, 1)  # (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, 0)  # (1, M, 4)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - inter

    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def giou_batch(bboxes1, bboxes2):
    """
    GIoU — generalized IoU, lepszy niż IoU przy braku nakładania się bboxów.
    Wartości w [-1, 1]; wyżej = lepiej.
    """
    b1 = np.expand_dims(bboxes1, 1)
    b2 = np.expand_dims(bboxes2, 0)

    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])

    w_inter = np.maximum(0., xx2 - xx1)
    h_inter = np.maximum(0., yy2 - yy1)
    inter = w_inter * h_inter

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter

    iou = np.where(union > 0, inter / union, 0.0)

    # Convex hull enclosing both boxes
    enc_x1 = np.minimum(b1[..., 0], b2[..., 0])
    enc_y1 = np.minimum(b1[..., 1], b2[..., 1])
    enc_x2 = np.maximum(b1[..., 2], b2[..., 2])
    enc_y2 = np.maximum(b1[..., 3], b2[..., 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - np.where(enc_area > 0, (enc_area - union) / enc_area, 0.0)
    return giou


def associate_detections_to_tracks(detections, track_bboxes, iou_threshold=0.3, use_giou=True, age_penalty=None):
    """
    Przypisuje detekcje do tracków przez Hungarian algorithm na macierzy (G)IoU.

    Returns:
        matches: (N, 2) array par (det_idx, trk_idx)
        unmatched_dets: indices detekcji bez przypisania
        unmatched_trks: indices tracków bez przypisania
    """
    if len(track_bboxes) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    # Cost matrix — używamy GIoU dla lepszej asocjacji przy częściowej okluzioni
    if use_giou:
        score_matrix = giou_batch(detections, track_bboxes)
    else:
        score_matrix = iou_batch(detections, track_bboxes)

    # Penalizuj tracki które długo nie były aktualizowane
    if age_penalty is not None and len(age_penalty) == score_matrix.shape[1]:
        score_matrix = score_matrix - age_penalty[np.newaxis, :]

    # Hungarian algorithm minimalizuje koszt → negujemy score
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    matches, unmatched_dets, unmatched_trks = [], [], []

    matched_set_d = set()
    matched_set_t = set()

    for r, c in zip(row_ind, col_ind):
        # GIoU > iou_threshold - 1 odpowiada IoU > iou_threshold dla zachowania spójności
        effective_threshold = iou_threshold - 1.0 if use_giou else iou_threshold
        if score_matrix[r, c] >= effective_threshold:
            matches.append([r, c])
            matched_set_d.add(r)
            matched_set_t.add(c)
        else:
            unmatched_dets.append(r)
            unmatched_trks.append(c)

    for d in range(len(detections)):
        if d not in matched_set_d:
            unmatched_dets.append(d)

    for t in range(len(track_bboxes)):
        if t not in matched_set_t:
            unmatched_trks.append(t)

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches, dtype=int)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)