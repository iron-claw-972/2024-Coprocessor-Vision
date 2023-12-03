import math
def get_x_offset_deg(self, box):
    #source: Limelight docs(LINK HERE)
    hfov = 62.8*(math.pi/180) 
    
    x1, y1, x2, y2 = box.xyxy[0]

    if len(box[0]):
        bbox_center_coord = [(x1+x2)/2,(y1+y2)/2]

        cx = bbox_center_coord[0]

        nx = (cx-320)/320

        vw = 2*math.tan((hfov/2))

        vx = vw/2*nx

        x_offset_deg = math.atan(vx/1)*(180/math.pi)

        return float(x_offset_deg)
                    
def get_y_offset_deg(self, det):
    #source: Limelight docs(LINK HERE)
    vfov = 37.9*(math.pi/180)

    if len(det):
        xyxy = det[:,:4][0]
        
        bbox_center_coord = [(xyxy[2]+xyxy[0])/2,(xyxy[3]+xyxy[1])/2]

        cy = bbox_center_coord[1]

        ny = (240-cy)/240 

        vh = 2*math.tan((vfov/2))

        vy = vh/2*ny

        y_offset_deg = math.atan(vy/1)*(180/math.pi)

        return float(y_offset_deg)    