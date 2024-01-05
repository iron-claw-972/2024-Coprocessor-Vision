import math

# TODO: Change these to the actual camera values
fov = [62.8, 37.9]

def get_x_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        hfov = fov[0]*(math.pi/180)
        x1, y1, x2, y2 = box.xyxy[0]

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cx = bbox_center_coord[0]

        nx = (cx-320)/320

        vw = 2*math.tan((hfov/2))

        vx = vw/2*nx

        x_offset_deg = math.atan(vx/1)*(180/math.pi)

        return float(x_offset_deg)
    
    return 0

def get_y_offset_deg(box):
    #source: Limelight docs(LINK HERE)

    if len(box[0]):
        vfov = fov[1]*(math.pi/180)
        x1, y1, x2, y2 = box.xyxy[0]

        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cy = bbox_center_coord[1]

        ny = (cy-320)/320

        vh = 2*math.tan((vfov/2))

        vy = vh/2*ny

        y_offset_deg = math.atan(vy/1)*(180/math.pi)

        return float(y_offset_deg)
    
    return 0
 
    
def get_distance(x_offset_deg,y_offset_deg,camera_height):
    x_offset_rad = x_offset_deg*(math.pi/180)
    y_offset_rad = y_offset_deg*(math.pi/180)

    distance_ideal = camera_height*math.tan(y_offset_rad)
    distance_actual = distance_ideal/math.cos(x_offset_rad)
    
    return distance_actual
