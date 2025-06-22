# position_tracker.py

import math

def extract_positions(results):
    positions = {
        "players": [],
        "goalkeepers": [],
        "main_referees": [],
        "side_referees": [],
        "staff_members": [],
        "ball": None
    }

    for box in results[0].boxes:
        try:
            cls_id = int(box.cls)
        except Exception:
            cls_id = int(box.cls[0])  # fallback for older format

        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])

        if cls_id == 0:  # player
            positions["players"].append((x, y))
        elif cls_id == 1:  # goalkeeper
            positions["goalkeepers"].append((x, y))
        elif cls_id == 2:  # ball
            positions["ball"] = (x, y)
        elif cls_id == 3:  # main referee
            positions["main_referees"].append((x, y))
        elif cls_id == 4:  # side referee
            positions["side_referees"].append((x, y))
        elif cls_id == 5:  # staff member
            positions["staff_members"].append((x, y))

    return positions

def find_ball_proximity(positions, threshold=50):
    ball_pos = positions.get("ball")
    if not ball_pos:
        return None, None

    nearest = None
    nearest_type = None
    min_dist = float('inf')

    for entity_type in ["players", "goalkeepers", "main_referees", "side_referees"]:
        for pos in positions.get(entity_type, []):
            dist = math.hypot(ball_pos[0] - pos[0], ball_pos[1] - pos[1])
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest = pos
                nearest_type = entity_type

    return nearest_type, min_dist if nearest else (None, None)

def count_detected_objects(results):
    """
    Returns a dictionary with counts for each class and ball presence.
    """
    counts = {
        "players": 0,
        "goalkeepers": 0,
        "main_referees": 0,
        "side_referees": 0,
        "staff_members": 0,
        "ball": 0
    }
    for box in results[0].boxes:
        try:
            cls_id = int(box.cls)
        except Exception:
            cls_id = int(box.cls[0])  # fallback for older format
        if cls_id == 0:
            counts["players"] += 1
        elif cls_id == 1:
            counts["goalkeepers"] += 1
        elif cls_id == 2:
            counts["ball"] += 1
        elif cls_id == 3:
            counts["main_referees"] += 1
        elif cls_id == 4:
            counts["side_referees"] += 1
        elif cls_id == 5:
            counts["staff_members"] += 1
    return counts
