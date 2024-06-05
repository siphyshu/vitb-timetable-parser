import json
import re

def parse_cell_info(cell_data):
    """
    Extract code, venue, and slot from the cell data.

    Args:
    - cell_data: Data extracted from a cell in the timetable.

    Returns:
    - Tuple containing slot code, class code, course type, and venue.
    """
    cell_data = cell_data.replace("\n", "")
    pattern = r"^(\w+)-(\w+)-(\w+)-(.*?)$"
    match = re.match(pattern, cell_data)

    if match:
        slot_code = match.group(1)
        class_code = match.group(2)
        course_type = match.group(3)
        venue = match.group(4).strip()
        
        return slot_code, class_code, course_type, venue

    
    return None, None, None, None


def tt2json(timetable):
    """
    Convert timetable data to JSON format.

    Args:
    - timetable: Extracted timetable data as a pandas DataFrame.

    Returns:
    - JSON representation of the timetable data.
    """
    days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    slot_buckets = {
        "1": ("08:30", "10:00"), 
        "2": ("10:05", "11:35"), 
        "3": ("11:40", "13:10"), 
        "4": ("13:15", "14:45"), 
        "5": ("14:50", "16:20"), 
        "6": ("16:25", "17:55"), 
        "7": ("18:00", "19:30")
    }

    timetable_json = {
        "days": []
    }

    if timetable is not None and len(timetable) == 0:
        raise ValueError("Timetable data is empty!")
    elif timetable is None:
        raise ValueError("Timetable data is empty!")

    # find the number of rows and columns in the timetable
    rows, cols = timetable.shape

    # correlate the days with the rows
    day_indices = {}
    for i in range(rows):
        for day in days:
            if day in timetable.iloc[i, 0]:
                day_indices[day] = i
                break

    # find start and end timing row indices by iterating over cells
    start_timing_row = None
    end_timing_row = None

    for i in range(rows):
        for j in range(cols):
            if timetable.iloc[i, j] == "Start":
                start_timing_row = i
            elif timetable.iloc[i, j] == "End":
                end_timing_row = i
            if start_timing_row is not None and end_timing_row is not None:
                break
        if start_timing_row is not None and end_timing_row is not None:
            break
    else:
        raise ValueError("Failed to find start and end timing rows in the timetable!")
    

    # extract slot indices from the timetable
    slot_indices = {}
    for i in range(cols):
        for slot_bucket, (start, end) in slot_buckets.items():
            if timetable.iloc[start_timing_row, i] == start and timetable.iloc[end_timing_row, i] == end:
                slot_indices[slot_bucket] = i
                break

    # go through the timetable and extract the slot data
    for day, index in day_indices.items():
        day_data = {
            "day": day,
            "classes": []
        }

        for slot_bucket, slot_index in slot_indices.items():
            cell_data = timetable.iloc[index, slot_index]
            slot_code, class_code, course_type, venue = parse_cell_info(cell_data)
            start, end = slot_buckets[slot_bucket]

            if class_code is not None and venue is not None and course_type is not None and slot_code is not None:
                day_data["classes"].append({
                    "slot": slot_code,
                    "class_code": class_code,
                    "course_type": course_type,
                    "venue": venue,
                    "start": start,
                    "end": end
                })

        timetable_json["days"].append(day_data)    

    return timetable_json