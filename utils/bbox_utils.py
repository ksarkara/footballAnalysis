def get_centre_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]                                     # x2 - x1

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5          # sqrt(x^2 + y^2)

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]                             # x2-x1, y2-y1

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)