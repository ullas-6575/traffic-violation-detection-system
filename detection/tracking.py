"""Occlusion-resilient vehicle tracker: ByteTrack + ghost-ID reassignment."""

from __future__ import annotations

import math

import supervision as sv


class OcclusionTracker:
    def __init__(self, frame_rate=30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate,
        )
        self.ghost_tracks       = {}
        self.id_map             = {}
        self.next_consistent_id = 1
        self.active_ids         = set()

    def update(self, detections, frame_num):
        tracked     = self.tracker.update_with_detections(detections)
        prev_active = self.active_ids.copy()
        self.active_ids = set()

        if tracked.tracker_id is None:
            return tracked, prev_active - self.active_ids

        new_ids = []
        for i, tid in enumerate(tracked.tracker_id.tolist()):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid in self.id_map:
                cid = self.id_map[tid]
            else:
                cid = self._try_reassign(cx, cy, frame_num)
                if cid is None:
                    cid = self.next_consistent_id
                    self.next_consistent_id += 1
                self.id_map[tid] = cid

            new_ids.append(cid)
            self.active_ids.add(cid)
            self.ghost_tracks[cid] = {
                "last_pos":   (float(cx), float(cy)),
                "lost_frame": int(frame_num),
            }

        self._clean_ghosts(frame_num)
        tracked.tracker_id = new_ids
        return tracked, prev_active - self.active_ids

    def _try_reassign(self, cx, cy, frame_num):
        best, best_dist = None, float("inf")
        for gid, g in self.ghost_tracks.items():
            if frame_num - g["lost_frame"] > 15:
                continue
            d = math.sqrt((float(cx) - g["last_pos"][0]) ** 2 +
                          (float(cy) - g["last_pos"][1]) ** 2)
            if d < 80 and d < best_dist:
                best_dist = d
                best      = gid
        if best is not None:
            del self.ghost_tracks[best]
            return best
        return None

    def _clean_ghosts(self, frame_num):
        stale = [gid for gid, g in self.ghost_tracks.items()
                 if frame_num - g["lost_frame"] > 15]
        for gid in stale:
            del self.ghost_tracks[gid]



