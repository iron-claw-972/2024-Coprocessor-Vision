import ntcore
import util
import time
from enum import Enum
from ultralytics.engine.results import Results # type: ignore

nt_inst = ntcore.NetworkTableInstance.getDefault()

pub_sub_options = ntcore.PubSubOptions(sendAll=True)

table = nt_inst.getTable("object_detection")
distance_topic = table.getDoubleArrayTopic("distance").publish(options=pub_sub_options)
x_angle_offset_topic = table.getDoubleArrayTopic("x_offset").publish(options=pub_sub_options)
y_angle_offset_topic = table.getDoubleArrayTopic("y_offset").publish(options=pub_sub_options)
object_class_topic = table.getStringArrayTopic("class").publish(options=pub_sub_options)
camera_index_topic = table.getIntegerArrayTopic("index").publish(options=pub_sub_options)
latency_topic = table.getDoubleArrayTopic("latency").publish(options=pub_sub_options)
flippy_topic = table.getBooleanTopic("flippy").publish(options=pub_sub_options)

class ObjClasses(Enum):
    ALGAE = 0
    CORAL = 1

#distance: list[float] = []
x_offset: list[float] = []
y_offset: list[float] = []
object_class: list[str] = []
camera_index: list[int] = []
latency_list: list[float] = []
last_flippy: bool = False

nt_inst.startClient4("Coprocessor")
nt_inst.setServerTeam(972)

def publish_distance() -> None:
    raise NotImplementedError("distance is not not implemented yet")
    #distance_topic.set(distance)

def publish_x_angle_offset() -> None:
    x_angle_offset_topic.set(x_offset)

def publish_y_angle_offset() -> None:
    y_angle_offset_topic.set(y_offset)

def publish_class() -> None:
    object_class_topic.set(object_class)

def publish_camera_index() -> None:
    camera_index_topic.set(camera_index)

def publish_latency() -> None:
    latency_topic.set(latency_list)

def flip_flippy() -> None:
    global last_flippy
    last_flippy = not last_flippy
    flippy_topic.set(last_flippy)

def add_results(results: list[Results], start_time: float, index: int) -> None:
    global nt_inst
    # Remove old values from the same camera, do not change values from other cameras
    i = len(camera_index) - 1
    while i >= 0:
        if camera_index[i] == index:
            x_offset.pop(i)
            y_offset.pop(i)
            #distance.pop(i)
            object_class.pop(i)
            camera_index.pop(i)
            latency_list.pop(i)
        i -= 1
    # Add new values to arrays: iterates over ALL detections, not just the first one
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        # Iterate over each individual detection in the result
        for box_idx in range(len(boxes)):
            xyxy = boxes.xyxy[box_idx]
            cls_id = int(boxes.cls[box_idx].cpu().numpy())
            x_offset.append(util.get_x_offset_deg_single(xyxy, boxes.orig_shape))
            y_offset.append(util.get_y_offset_deg_single(xyxy, boxes.orig_shape))
            #distance.append(util.get_distance(box))
            object_class.append(result.names[cls_id])
            camera_index.append(index)
            latency_list.append(time.time() - start_time)
    # Publish values to NetworkTables
    #publish_distance()
    publish_x_angle_offset()
    publish_y_angle_offset()
    publish_class()
    publish_camera_index()
    publish_latency()
    flip_flippy()
    nt_inst.flush()
    
