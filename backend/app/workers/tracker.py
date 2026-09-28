import time

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class HumanTracker:
    def __init__(self, iou_threshold=0.3, max_missing_frames=15):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames

    def update(self, detections):
        new_tracks = {}
        matched_detects = set()
        matched_tracks = set()

        for track_id, track_data in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_detects:
                    continue
                iou = calculate_iou(track_data["box"], det)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx != -1:
                new_tracks[track_id] = {
                    "box": detections[best_det_idx],
                    "missing_frames": 0,
                    "db_event_id": track_data.get("db_event_id")
                }
                matched_detects.add(best_det_idx)
                matched_tracks.add(track_id)
            else:
                if track_data["missing_frames"] < self.max_missing_frames:
                    new_tracks[track_id] = {
                        "box": track_data["box"],
                        "missing_frames": track_data["missing_frames"] + 1,
                        "db_event_id": track_data.get("db_event_id")
                    }

        lost_tracks = [t_data for t_id, t_data in self.tracks.items() if t_id not in new_tracks]

        newly_created = []
        for i, det in enumerate(detections):
            if i not in matched_detects:
                new_tracks[self.next_id] = {
                    "box": det,
                    "missing_frames": 0,
                    "db_event_id": None 
                }
                newly_created.append(self.next_id)
                self.next_id += 1

        self.tracks = new_tracks
        
        return newly_created, [t["db_event_id"] for t in lost_tracks if t["db_event_id"] is not None]
        
    def set_db_event_id(self, track_id, db_event_id):
        if track_id in self.tracks:
            self.tracks[track_id]["db_event_id"] = db_event_id
