import numpy as np
import math as mt
import sys
import pdb

PHI = 3.141592

# "keypoints": [ 0"nose", 1"left_eye", 2"right_eye", 3"left_ear", 4"right_ear",
# 5"left_shoulder", 6"right_shoulder", 7"left_elbow", 8"right_elbow", 9"left_wrist",
# 10"right_wrist", 11"left_hip", 12"right_hip", 13"left_knee", 14"right_knee",
# 15"left_ankle", 16"right_ankle" ]

def get_distance(a, b):

    #a,b -> list composed of coordinate x, y

    a_x = a[0]
    a_y = a[1]
    b_x = b[0]
    b_y = b[1]

    result = mt.sqrt(mt.pow((a_x - b_x), 2) + pow((a_y - b_y), 2))
    if result == 0:
        result = sys.float_info.epsilon()

    return result

def ladder_error_detection(pose):
    pose = np.array(pose)
    wrist_l = pose[9]
    wrist_r = pose[10]
    hip_l = pose[11]
    hip_r = pose[12]
    wrist_hip_l = get_distance(wrist_l, hip_l)
    wrist_hip_r = get_distance(wrist_r, hip_r)

    dist_threshold = 50

    if wrist_hip_l < dist_threshold or wrist_hip_r < dist_threshold:
        ladder_error_code = True

    return ladder_error_code

def lift_error_detection(pose):
    pose = np.array(pose)
    shoulder_l = pose[5]
    shoulder_height = pose[5][1]
    hip_l = pose[11]
    hip_height = pose[11][1]
    knee_l = pose[13]
    knee_height = pose[13][1]

    angle_threshold = 50
    hip_height_threshold = 50

    shoulder_hip = get_distance(shoulder_l, hip_l)
    hip_knee = get_distance(hip_l, knee_l)
    shoulder_knee = get_distance(shoulder_l, knee_l)

    angle_error = mt.cos(
        (mt.pow(shoulder_hip, 2) + mt.pow(hip_knee, 2) - mt.pow(shoulder_knee, 2)) /
        (2 * shoulder_hip * hip_knee)
    )
    angle_error = (angle_error * 180) / PHI

    hip_height_error = abs(hip_height - knee_height)

    if angle_error < angle_threshold or hip_height_error <hip_height_threshold:
        list_error_code = True

    return list_error_code

def saw_error_detection(pose):
    pose = np.array(pose)
    ankle_l = pose[15]
    ankle_r = pose[16]
    ankle_l_x = pose[15][0]
    ankle_r_x = pose[16][0]

    dist_threshold = 50

    dist_error = abs(ankle_r_x - ankle_l_x)

    if dist_error < dist_threshold:
        saw_error_code = True

    return saw_error_code

def shovel_error_detection(pose):
    pose = np.array(pose)
    wrist_l = pose[9]
    wrist_r = pose[10]
    ankle_l = pose[15]
    ankle_r = pose[16]
    knee_l = pose[13]
    knee_r = pose[14]
    hip_l = pose[11]
    hip_r = pose[12]
    
    dist_threshold = 50
    angle_threshold = 50
    
    dist_error_wrist = get_distance(wrist_l, wrist_r)
    dist_error_ankle = get_distance(ankle_l, ankle_r)
    hip_knee_l = get_distance(hip_l, knee_l)
    knee_ankle_l = get_distance(knee_l, ankle_l)
    hip_ankle_l = get_distance(hip_l, ankle_l))
    hip_knee_r = get_distance(hip_r, knee_r)
    knee_ankle_r = get_distance(knee_r, ankle_r)
    hip_ankle_r = get_distance(hip_r, ankle_r)
    
    angle_error_l = mt.cos(
        (mt.pow(hip_knee_l, 2) + mt.pow(knee_ankle_l, 2) - mt.pow(hip_ankle_l, 2)) /
        (2 * hip_knee_l * knee_ankle_l)
        )
    angle_error_l = (angle_error_l * 180) / PHI
    angle_error_r = mt.cos(
        (mt.pow(hip_knee_r, 2) + mt.pow(knee_ankle_r, 2) - mt.pow(hip_ankle_r, 2)) /
        (2 * hip_knee_r * knee_ankle_r)
        )
    angle_error_r = (angle_error_r * 180) / PHI
    
    if dist_error_wrist < dist_threshold or dist_error_ankle < dist_threshold or angle_error_l < angle_threshold or angle_error_r < angle_threshold:
        shovel_error_code = True

    return shovel_error_code
    
def walk_error_detection(pose):
    pose = np.array(pose)
    wrist_l = pose[9]
    wrist_r = pose[10]
    hip_l = pose[11]
    hip_r = pose[12]
    
    dist_threshold = 20
    dist_error_l = get_distance(wrist_l, hip_l)
    dist_error_r - get_distance(wrist_r, hip_r)
    
    if dist_error_l < dist_threshold or dist_error_r < dist_threshold:
        walk_error_code = True

    return walk_error_code
