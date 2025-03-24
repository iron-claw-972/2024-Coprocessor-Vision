# import math
# from ultralytics.engine.results import Boxes # type: ignore

# # TODO: Change these to the actual camera values
# fov = [70, 43.75] # Arducam
# RESOLUTION = [720, 1280]
# # fov = [59.703, 33.583] # Microsoft lifecam or other cameras with diagonal FOV of 68.5 degrees and 1280x720 resolution

# def get_fovs(box: Boxes) -> list[float]:
#     x1, y1, x2, y2 = box.xyxy[0]
#     center: list[float] = [(x1+x2)/2,(y1+y2)/2]
#     zero_centered: list[float] = [center[0] - (RESOLUTION[0]/2), center[1] - (RESOLUTION[1]/2)]
#     viewport: list[float] = [zero_centered[0], zero_centered[1], 1]
#     fovs: list[float] = [math.tan(viewport[0]) * (fov[0]/2), math.tan(viewport[1]) * (fov[1]/2)]
#     return fovs
# z
# def get_x_offset_deg(box: Boxes) -> float:
#     #source: Limelight docs(LINK HERE)

#     if len(box[0]):
#         hfov = fov[0]*(math.pi/180)
#         x1, y1, x2, y2 = box.xyxy[0]

#         bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

#         cx = bbox_center_coord[0]

#         nx = (cx-320)/320

#         vw = 2*math.tan((hfov/2))

#         vx = vw/2*nx

#         x_offset_deg = math.atan(vx/1)*(180/math.pi)

#         return float(x_offset_deg)
    
#     return 0

# def get_y_offset_deg(box: Boxes) -> float:
#     #source: Limelight docs(LINK HERE)

#     if len(box[0]):
#         vfov = fov[1]*(math.pi/180)
#         x1, y1, x2, y2 = box.xyxy[0]

#         bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

#         cy = bbox_center_coord[1]

#         ny = (cy-320)/320

#         vh = 2*math.tan((vfov/2))

#         vy = vh/2*ny

#         y_offset_deg = math.atan(vy/1)*(180/math.pi)

#         return float(y_offset_deg)
    
#     return 0
 
    
# # TODO: Add tis if we're using it, low pirority
# def get_distance(box: Boxes) -> float:
#     raise NotImplementedError("get_distance doesn't exist yet -- please write it!")


import math
from ultralytics.engine.results import Boxes  # type: ignore

# TODO: Change these to the actual camera values
fov = [70, 43.75]  # Arducam
RESOLUTION = [1280, 720]  # [Height, Width] (Fixed order)

# fov = [59.703, 33.583]  # Microsoft lifecam or other cameras with diagonal FOV of 68.5 degrees and 1280x720 resolution


def get_fovs(box: Boxes) -> list[float]:
    if len(box) == 0:
        return [0.0, 0.0]

    x1, y1, x2, y2 = box.xyxy[0]
    center = [(x1 + x2) / 2, (y1 + y2) / 2]

    # Normalize center coordinates
    zero_centered = [
        (center[0] - (RESOLUTION[1] / 2)) / (RESOLUTION[1] / 2),  # Normalized x
        (center[1] - (RESOLUTION[0] / 2)) / (RESOLUTION[0] / 2)   # Normalized y
    ]

    # Convert to field-of-view angles
    fovs = [
        math.degrees(math.atan(zero_centered[0] * math.tan(math.radians(fov[0] / 2)))),
        math.degrees(math.atan(zero_centered[1] * math.tan(math.radians(fov[1] / 2))))
    ]
    return fovs


def get_x_offset_deg(box: Boxes) -> float:
    if len(box) == 0:
        return 0.0

    hfov = math.radians(fov[0])
    x1, _, x2, _ = box.xyxy[0]

    bbox_center_x = (x1 + x2) / 2
    normalized_x = (bbox_center_x - (RESOLUTION[1] / 2)) / (RESOLUTION[1] / 2)

    vw = 2 * math.tan(hfov / 2)
    vx = vw / 2 * normalized_x

    x_offset_deg = math.degrees(math.atan(vx))
    return x_offset_deg


def get_y_offset_deg(box: Boxes) -> float:
    if len(box) == 0:
        return 0.0

    vfov = math.radians(fov[1])
    _, y1, _, y2 = box.xyxy[0]

    bbox_center_y = (y1 + y2) / 2
    normalized_y = (bbox_center_y - (RESOLUTION[0] / 2)) / (RESOLUTION[0] / 2)

    vh = 2 * math.tan(vfov / 2)
    vy = vh / 2 * normalized_y

    y_offset_deg = math.degrees(math.atan(vy))
    return y_offset_deg


# TODO: Add this if we're using it, low priority
def get_distance(box: Boxes) -> float:
    raise NotImplementedError("get_distance doesn't exist yet -- please write it!")
