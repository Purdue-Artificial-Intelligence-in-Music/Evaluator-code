import json
import os
import threading
import time
from datetime import datetime


class Profile:
    _ts = ""
    _finalized_detail_json = {}
    _finalized_summary_json = {}

    @classmethod
    def get_timestamp(cls):
        return cls._ts

    @classmethod
    def get_detail_json(cls, video_name):
        return cls._finalized_detail_json.get(video_name)

    @classmethod
    def get_summary_json(cls, video_name):
        return cls._finalized_summary_json.get(video_name)

    def __init__(self, output_dir="sessions"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._session_data = {}
        self._output_files = {}
        self._session_timestamps_formatted = {}
        self._session_start_times = {}

        self._accumulated_height = {}
        self._accumulated_angle = {}
        self._accumulated_hand = {}
        self._accumulated_pose = {}
        self._accumulated_hand_posture = {}
        self._accumulated_elbow_posture = {}

        self._timers = {}

    def create_session(self, video_name):
        if video_name in self._session_data:
            return

        self._session_data[video_name] = []

        now = datetime.now()
        Profile._ts = now.strftime("%Y%m%d_%H%M%S")

        self._session_timestamps_formatted[video_name] = now.strftime("%Y-%m-%d %H:%M:%S")
        self._session_start_times[video_name] = time.time()

        self._accumulated_height[video_name] = {"Top": 0, "Middle": 0, "Bottom": 0, "Outside": 0, "Unknown": 0}
        self._accumulated_angle[video_name] = {"Correct": 0, "Wrong": 0, "Unknown": 0}
        self._accumulated_hand[video_name] = {"Detected": 0, "None": 0}
        self._accumulated_pose[video_name] = {"Detected": 0, "None": 0}
        self._accumulated_hand_posture[video_name] = {}
        self._accumulated_elbow_posture[video_name] = {}

        filepath = os.path.join(self.output_dir, f"session_{video_name}.json")
        with open(filepath, 'w') as f:
            f.write(f'{{"video_name":"{video_name}","data":[')
        self._output_files[video_name] = filepath

        self._schedule_breakdown(video_name)

    def _schedule_breakdown(self, video_name):
        def tick():
            if video_name in self._session_data:
                self._append_new_breakdown(video_name)
                self._schedule_breakdown(video_name)

        timer = threading.Timer(10.0, tick)
        timer.daemon = True
        timer.start()
        self._timers[video_name] = timer

    def add_session_data(self, video_name, bow_result=None, hand_result=None):
        """
        bow_result  : dict from Classification.process_frame
                      {"class": int, "bow": list, "string": list, "angle": int}
        hand_result : dict from Hands.process_frame
                      {"hand_class": str, "handedness": str, "brect": list,
                       "landmark_list": list, "elbow_class": str, "pose_points": list}
        """
        if video_name not in self._session_data:
            self.create_session(video_name)
        self._session_data[video_name].append({
            "bow": bow_result,
            "hand": hand_result,
        })

    def _append_new_breakdown(self, video_name):
        session = self._session_data.get(video_name)
        if not session:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        window = self._analyze_session_window(session, timestamp, video_name)
        filepath = self._output_files.get(video_name)
        if filepath:
            with open(filepath, 'a') as f:
                f.write(json.dumps(window) + ",")

        self._session_data[video_name].clear()

    def _analyze_session_window(self, session, timestamp, video_name):
        bow_frames = [d["bow"] for d in session if d.get("bow") is not None]
        hand_frames = [d["hand"] for d in session if d.get("hand") is not None]

        height_breakdown = {}
        angle_breakdown = {}

        if bow_frames:
            total = len(bow_frames)
            height_counts = {"Top": 0, "Middle": 0, "Bottom": 0, "Outside": 0, "Unknown": 0}
            angle_counts = {"Correct": 0, "Wrong": 0, "Unknown": 0}

            for frame in bow_frames:
                cls = frame.get("class")
                if cls == 2:
                    height_counts["Top"] += 1
                elif cls == 0:
                    height_counts["Middle"] += 1
                elif cls == 3:
                    height_counts["Bottom"] += 1
                elif cls == 1:
                    height_counts["Outside"] += 1
                else:
                    height_counts["Unknown"] += 1

                angle = frame.get("angle")
                if angle == 0:
                    angle_counts["Correct"] += 1
                elif angle == 1:
                    angle_counts["Wrong"] += 1
                else:
                    angle_counts["Unknown"] += 1

            for k, v in height_counts.items():
                self._accumulated_height[video_name][k] += v
            for k, v in angle_counts.items():
                self._accumulated_angle[video_name][k] += v

            height_breakdown = {k: (v / total) * 100 for k, v in height_counts.items()}
            angle_breakdown = {k: (v / total) * 100 for k, v in angle_counts.items()}

        hand_presence = {}
        hand_posture = {}
        pose_presence = {}
        elbow_posture = {}

        if hand_frames:
            total = len(hand_frames)
            hand_counts = {"Detected": 0, "None": 0}
            pose_counts = {"Detected": 0, "None": 0}
            hand_posture_counts = {}
            elbow_posture_counts = {}

            hand_label_map = {0: "Correct", 1: "Supination", 2: "Too much pronation"}
            elbow_label_map = {0: "Correct", 1: "Low elbow", 2: "Elbow too high"}

            for frame in hand_frames:
                if frame.get("hand_class") is not None:
                    hand_counts["Detected"] += 1
                    hand_id = frame.get("hand_class_id")
                    label = hand_label_map.get(hand_id, "Unknown")
                    hand_posture_counts[label] = hand_posture_counts.get(label, 0) + 1
                else:
                    hand_counts["None"] += 1

                if frame.get("elbow_class") is not None:
                    pose_counts["Detected"] += 1
                    elbow_id = frame.get("elbow_class_id")
                    label = elbow_label_map.get(elbow_id, "Unknown")
                    elbow_posture_counts[label] = elbow_posture_counts.get(label, 0) + 1
                else:
                    pose_counts["None"] += 1

            for k, v in hand_counts.items():
                self._accumulated_hand[video_name][k] += v
            for k, v in pose_counts.items():
                self._accumulated_pose[video_name][k] += v
            for k, v in hand_posture_counts.items():
                self._accumulated_hand_posture[video_name][k] = \
                    self._accumulated_hand_posture[video_name].get(k, 0) + v
            for k, v in elbow_posture_counts.items():
                self._accumulated_elbow_posture[video_name][k] = \
                    self._accumulated_elbow_posture[video_name].get(k, 0) + v

            hand_presence = {k: (v / total) * 100 for k, v in hand_counts.items()}
            pose_presence = {k: (v / total) * 100 for k, v in pose_counts.items()}
            hand_posture = {k: (v / total) * 100 for k, v in hand_posture_counts.items()}
            elbow_posture = {k: (v / total) * 100 for k, v in elbow_posture_counts.items()}

        return {
            "video_name": video_name,
            "timestamp": timestamp,
            "durationSeconds": 0,
            "durationFormatted": "0s",
            "heightBreakdown": height_breakdown,
            "angleBreakdown": angle_breakdown,
            "handPresenceBreakdown": hand_presence,
            "handPostureBreakdown": hand_posture,
            "posePresenceBreakdown": pose_presence,
            "elbowPostureBreakdown": elbow_posture,
        }

    def end_session_and_get_summary(self, video_name):
        timer = self._timers.pop(video_name, None)
        if timer:
            timer.cancel()

        session = self._session_data.get(video_name)
        filepath = self._output_files.get(video_name)

        if session:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            window = self._analyze_session_window(session, timestamp, video_name)
            if filepath:
                with open(filepath, 'a') as f:
                    f.write(json.dumps(window) + ",")

        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
            if content.endswith(","):
                content = content[:-1]
            content += "]}"
            with open(filepath, 'w') as f:
                f.write(content)
            Profile._finalized_detail_json[video_name] = content

        start_time = self._session_start_times.get(video_name, time.time())
        duration_s = int(time.time() - start_time)
        duration_formatted = self._format_duration(duration_s)

        session_timestamp = self._session_timestamps_formatted.get(
            video_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        total_summary = self._generate_total_summary(
            video_name, session_timestamp, duration_s, duration_formatted
        )

        summary_path = os.path.join(self.output_dir, f"session_{video_name}_summary.json")
        summary_json = json.dumps(total_summary, indent=2)
        with open(summary_path, 'w') as f:
            f.write(summary_json)
        Profile._finalized_summary_json[video_name] = summary_json

        # Cleanup
        for store in (self._session_data, self._output_files,
                      self._session_timestamps_formatted,
                      self._session_start_times, self._accumulated_height,
                      self._accumulated_angle, self._accumulated_hand,
                      self._accumulated_pose, self._accumulated_hand_posture,
                      self._accumulated_elbow_posture):
            store.pop(video_name, None)

        return total_summary

    def _generate_total_summary(self, video_name, timestamp, duration_seconds, duration_formatted):
        def to_pct(counts):
            total = sum(counts.values())
            if total == 0:
                return {}
            return {k: (v / total) * 100 for k, v in counts.items()}

        return {
            "video_name": video_name,
            "timestamp": timestamp,
            "durationSeconds": duration_seconds,
            "durationFormatted": duration_formatted,
            "heightBreakdown": to_pct(self._accumulated_height.get(video_name, {})),
            "angleBreakdown": to_pct(self._accumulated_angle.get(video_name, {})),
            "handPresenceBreakdown": to_pct(self._accumulated_hand.get(video_name, {})),
            "handPostureBreakdown": to_pct(self._accumulated_hand_posture.get(video_name, {})),
            "posePresenceBreakdown": to_pct(self._accumulated_pose.get(video_name, {})),
            "elbowPostureBreakdown": to_pct(self._accumulated_elbow_posture.get(video_name, {})),
        }

    @staticmethod
    def _format_duration(seconds):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"
