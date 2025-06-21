# position_tracker.py

def extract_positions(results):
    positions = {
        "players": [],
        "goalkeepers": [],
        "referees": [],
        "ball": None
    }

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])

        if cls_id == 0:  # ball
            positions["ball"] = (x, y)
        elif cls_id == 1:  # goalkeeper
            positions["goalkeepers"].append((x, y))
        elif cls_id == 2:  # player
            positions["players"].append((x, y))
        elif cls_id == 3:  # referee
            positions["referees"].append((x, y))

    return positions
