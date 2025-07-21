#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""

import carla
import argparse
import random
import time
import numpy as np
from numpy import random
import open3d as o3d
import matplotlib
import math
import copy
import os
from scipy.spatial.transform import Rotation as R


vehicle_ego = None
vehicle_veh = None
vehicle_gap = None
SYNC_OUT_DIR = "_out"
def extract_yaw_from_open3d_R(R_mat):
    """
    Open3D의 3x3 회전행렬로부터 Yaw (Z축 회전) 추출 (deg)
    """
    r = R.from_matrix(R_mat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return yaw


#gpt
def get_transformation_matrix(location, rotation):
    """
    CARLA Location + Rotation → 4x4 Homogeneous Transform (좌수 → 우수 변환 포함)
    """
    tx, ty, tz = location.x, location.y, location.z
    roll = np.deg2rad(rotation.roll)
    pitch = np.deg2rad(rotation.pitch)
    yaw = np.deg2rad(rotation.yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 좌수 → 우수 변환 (Y축 플립)
    R_mat = Rz @ Ry @ Rx
    R_mat[:, 1] *= -1
    R_mat[1, :] *= -1

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = np.array([tx, ty, tz])

    return T



#gpt
# def carla_bbox_to_open3d_obb(bbox, vehicle_transform, lidar_transform):
#     """
#     CARLA BoundingBox → LiDAR 좌표계 기준 Open3D OBB 변환
#     """
#     # 1. LiDAR의 World→Sensor 변환행렬 (4x4)
#     T_world_to_lidar = np.linalg.inv(get_transformation_matrix(lidar_transform.location, lidar_transform.rotation))

#     # 2. BBox center (world 기준)
#     bbox_world_center = vehicle_transform.transform(bbox.location)
#     bbox_center_world = np.array([
#         bbox_world_center.x,
#         bbox_world_center.y,
#         bbox_world_center.z,
#         1.0
#     ])

#     # 3. BBox center → LiDAR frame
#     bbox_center_lidar_h = T_world_to_lidar @ bbox_center_world
#     bbox_center_lidar = bbox_center_lidar_h[:3]

#     # 4. 차량 회전행렬 (Open3D 우수 기준)
#     R_vehicle_world = convert_carla_rotation_to_open3d(vehicle_transform.rotation)

#     # 5. 차량 회전 → LiDAR frame으로 변환
#     R_lidar_vehicle = T_world_to_lidar[:3, :3] @ R_vehicle_world

#     # 6. 좌수→우수 변환 보정 (X축 Flip)
#     bbox_center_lidar[0] *= -1
#     R_lidar_vehicle[:, 0] *= -1

#     # 7. Left-handed 확인 후 보정 (det<0일 경우 Y축도 추가 플립)
#     det = np.linalg.det(R_lidar_vehicle)
#     if det < 0:
#         print("[경고] Left-handed 회전행렬! Y축 추가 플립")
#         R_lidar_vehicle[:, 1] *= -1

#     # 8. BBox 크기 설정 (full size)
#     extent = bbox.extent
#     bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2])

#     # 9. Open3D OBB 최종 생성
#     obb = o3d.geometry.OrientedBoundingBox(
#         center=bbox_center_lidar,
#         R=R_lidar_vehicle,
#         extent=bbox_extent
#     )
#     obb.color = (1, 0, 0)  # 빨간색 (원하는대로 변경 가능)

#     return obb





# def carla_rotation_to_matrix(rotation):
#     """CARLA의 Yaw만 반영한 회전 행렬 (Z축 회전)"""
#     yaw = math.radians(-rotation.yaw)
#     cos_yaw = math.cos(yaw)
#     sin_yaw = math.sin(yaw)
#     return np.array([
#         [cos_yaw, -sin_yaw, 0],
#         [sin_yaw,  cos_yaw, 0],
#         [0,        0,       1]
#     ])

# def convert_carla_rotation_to_open3d(rotation: carla.Rotation) -> np.ndarray:
#     """CARLA 회전 (Yaw/Pitch/Roll)을 Open3D 우수계 회전 행렬로 변환"""
#     yaw = -np.radians(rotation.yaw)
#     pitch = -np.radians(rotation.pitch)
#     roll = np.radians(rotation.roll)

#     R_yaw = np.array([
#         [np.cos(yaw), -np.sin(yaw), 0],
#         [np.sin(yaw),  np.cos(yaw), 0],
#         [0, 0, 1]
#     ])
#     R_pitch = np.array([
#         [np.cos(pitch), 0, np.sin(pitch)],
#         [0, 1, 0],
#         [-np.sin(pitch), 0, np.cos(pitch)]
#     ])
#     R_roll = np.array([
#         [1, 0, 0],
#         [0, np.cos(roll), -np.sin(roll)],
#         [0, np.sin(roll), np.cos(roll)]
#     ])

#     R = R_yaw @ R_pitch @ R_roll

#     # 좌수계(CARLA) → 우수계(Open3D) 변환: Y축 기준 반전
#     R[:, 1] *= -1
#     R[1, :] *= -1
#     return R


# 원본

# def create_bbox_from_carla(bbox, vehicle_transform, lidar_transform):
#     # 1. bbox 중심 (world 좌표계 기준)
#     bbox_world_center = vehicle_transform.transform(bbox.location)
#     bbox_center_world = np.array([
#         bbox_world_center.x,
#         bbox_world_center.y,
#         bbox_world_center.z,
#         1.0
#     ])

#     # 2. 차량 회전행렬 (world 기준)
#     #R_vehicle = carla_rotation_to_matrix(vehicle_transform.rotation)
#     R_vehicle = convert_carla_rotation_to_open3d(vehicle_transform.rotation)
#     # 3. LiDAR의 pose (world 기준)
#     T_world_to_lidar = np.linalg.inv(np.array(lidar_transform.get_matrix()))

#     # 4. 바운딩 박스 중심 변환 (world → lidar)
#     bbox_center_lidar = T_world_to_lidar @ bbox_center_world
#     bbox_center_lidar = bbox_center_lidar[:3]
#     #bbox_center_lidar[1] *= -1  # 좌수계 → 우수계 변환

#     # 5. 차량 회전행렬 → lidar 기준
#     R_world = np.eye(4)
#     R_world[:3, :3] = R_vehicle
#     R_lidar = T_world_to_lidar @ R_world @ np.linalg.inv(T_world_to_lidar)
#     R_lidar = R_lidar[:3, :3]

#     # 6. 회전 좌표계 반전 적용
#     #R_lidar[:, 1] *= -1  # Y축 반전
#     #R_lidar[1, :] *= -1

#     # 7. 바운딩 박스 크기 (full extent)
#     extent = bbox.extent
#     bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2]) # 기존 x,y
#     bbox_center_lidar[0] = -bbox_center_lidar[0]
#     #bbox_center_lidar[1] = -bbox_center_lidar[1]
#     # 8. Open3D OBB 객체 생성
#     bbox_o3d = o3d.geometry.OrientedBoundingBox(bbox_center_lidar,R_lidar,bbox_extent )
#     #bbox_o3d.center = bbox_center_lidar
#     #bbox_o3d.center[1] = -bbox_o3d.center[1]
#     #bbox_o3d.extent = bbox_extent
#     #bbox_o3d.R = R_lidar
#     bbox_o3d.color=(0,0,1)

#     return bbox_o3d



# def convert_carla_rotation_to_open3d(rotation: carla.Rotation) -> np.ndarray:
#     """
#     CARLA 좌수좌표계의 roll, pitch, yaw를 Open3D 우수좌표계 기준의 회전 행렬로 변환
#     ZYX (yaw → pitch → roll) 순서로 회전
#     """
#     yaw = np.radians(rotation.yaw)
#     pitch = np.radians(rotation.pitch)
#     roll = np.radians(rotation.roll)
#     rot = R.from_euler('ZYX', [yaw, pitch, roll])
#     return rot.as_matrix()  # (3,3)



def convert_carla_rotation_to_open3d(rotation):
    """
    CARLA의 carla.Rotation → 회전 행렬 (Open3D 호환)
    """
    pitch = np.deg2rad(rotation.pitch)
    yaw = np.deg2rad(rotation.yaw)
    roll = np.deg2rad(rotation.roll)

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_roll = np.array([
        [1, 0,           0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    # 순서: Roll → Pitch → Yaw
    return R_yaw @ R_pitch @ R_roll





# ego차량 잘 그려지는 원본
# def create_bbox_from_carla(bbox: carla.BoundingBox,
#                             vehicle_transform: carla.Transform,
#                             lidar_transform: carla.Transform) -> o3d.geometry.OrientedBoundingBox:
#     """
#     CARLA vehicle의 BoundingBox를 LiDAR 좌표계 기준 Open3D OBB로 변환
#     """
#     # 1. 바운딩 박스 중심 (월드 기준)
#     bbox_world_center = vehicle_transform.transform(bbox.location)
#     bbox_center_world = np.array([
#         bbox_world_center.x,
#         bbox_world_center.y,
#         bbox_world_center.z
#     ])

#     # 2. 차량 회전 행렬 (Open3D 우수 기준)
#     R_vehicle = convert_carla_rotation_to_open3d(vehicle_transform.rotation)  # (3x3)

#     # 3. LiDAR의 월드 기준 변환행렬
#     T_world_to_lidar = np.linalg.inv(np.array(lidar_transform.get_matrix()))  # (4x4)

#     # 4. bbox 중심 → lidar 기준으로 변환
#     bbox_center_world_h = np.append(bbox_center_world, 1.0)  # (4,)
#     bbox_center_lidar_h = T_world_to_lidar @ bbox_center_world_h
#     bbox_center_lidar = bbox_center_lidar_h[:3]

#     # 5. 차량 회전 → lidar 좌표계로 변환 (회전만 적용)
#     R_lidar = T_world_to_lidar[:3, :3] @ R_vehicle

#     # 6. 좌표계 반전 보정 (CARLA 좌수 → Open3D 우수)
#     bbox_center_lidar[0] *= -1          # X 좌표 반전
#     R_lidar[:, 0] *= -1                 # X축 방향 반전
#     #R_lidar[:, 1] *= -1                 # X축 방향 반전
    
#     # 7. 바운딩 박스 크기 (extent는 반길이 → full size)
#     extent = bbox.extent
#     bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2])

#     # 8. Open3D OBB 생성
#     obb = o3d.geometry.OrientedBoundingBox(center=bbox_center_lidar,
#                                            R=R_lidar,
#                                            extent=bbox_extent)
#     obb.color = (0, 0, 1)  # 파란색
#     #R = obb.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
#     #obb.rotate(R, center=obb.get_center())
#     return obb




#ego차량은 잘 작동되는 bbox

# def create_bbox_from_carla(bbox, vehicle_transform, lidar_transform):
#     """
#     특정 차량의 BoundingBox를 현재 Ego LiDAR 좌표계 기준으로 변환
#     """
#     # 1. 차량 bbox center (world 기준)
#     bbox_world_center = vehicle_transform.transform(bbox.location)
#     bbox_center_world = np.array([
#         bbox_world_center.x,
#         bbox_world_center.y,
#         bbox_world_center.z,
#         1.0
#     ])

#     # 2. 차량 회전행렬 (CARLA → Open3D 우수)
#     R_vehicle = convert_carla_rotation_to_open3d(vehicle_transform.rotation)  # (3x3)

#     # 3. World → LiDAR 변환 행렬
#     T_world_to_lidar = np.linalg.inv(np.array(lidar_transform.get_matrix()))  # (4x4)

#     # 4. bbox center 변환
#     bbox_center_lidar_h = T_world_to_lidar @ bbox_center_world
#     bbox_center_lidar = bbox_center_lidar_h[:3]

#     # 5. 차량 회전행렬도 LiDAR 기준으로
#     R_lidar = T_world_to_lidar[:3, :3] @ R_vehicle

#     # 6. 좌수 → 우수 변환 (X축 플립)
#     bbox_center_lidar[0] *= -1
#     R_lidar[:, 1] *= -1

#     # 7. Left-handed 행렬 보정 (det < 0 이면 Y축 추가 플립)
#     if np.linalg.det(R_lidar) < 0:
#         R_lidar[:, 1] *= -1

#     # 8. BoundingBox 크기
#     extent = bbox.extent
#     bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2])

#     # 9. Open3D OBB 생성
#     obb = o3d.geometry.OrientedBoundingBox(center=bbox_center_lidar,
#                                            R=R_lidar,
#                                            extent=bbox_extent)
#     return obb


#gpt피셜
def create_bbox_from_actor(actor, lidar_transform, ego_transform, is_ego=False):
    bbox = actor.bounding_box
    vehicle_tf = actor.get_transform()
    
    # 1. World → LiDAR 변환 행렬
    world2lidar = np.linalg.inv(np.array(lidar_transform.get_matrix()))
    
    # 2. 바운딩 박스 중심 (World 기준)
    bbox_world = vehicle_tf.transform(bbox.location)
    bbox_center_world = np.array([bbox_world.x, bbox_world.y, bbox_world.z, 1.0])
    
    # 3. World → LiDAR로 변환 (X축 반전 적용)
    bbox_center_lidar_h = world2lidar @ bbox_center_world
    bbox_center_lidar = bbox_center_lidar_h[:3]
    bbox_center_lidar[0] *= -1  # X축 반전: LiDAR X축이 후방을 가리킴을 보정
    
    # 4. 차량 회전 행렬 (CARLA → Open3D)
    R_vehicle = convert_carla_rotation_to_open3d(vehicle_tf.rotation)
    
    if is_ego:
        bbox_center_lidar = np.array([0.0, 0.0, 0.0])
        R_lidar = np.eye(3)
    else:
        ego_yaw_rad = np.radians(ego_transform.rotation.yaw)
        vehicle_yaw_rad = np.radians(vehicle_tf.rotation.yaw)
        correction_rad = vehicle_yaw_rad - ego_yaw_rad  # 차량의 yaw를 ego 기준으로
        if abs(correction_rad) > np.pi:
            correction_rad = (correction_rad + np.pi) % (2 * np.pi) - np.pi  # -180도~180도 범위로 정규화
        correction_R = R.from_euler('z', -correction_rad).as_matrix()
        # world2lidar의 회전 제거 후 보정 적용
        R_lidar = correction_R @ R_vehicle  # world2lidar의 회전 제외
        if np.linalg.det(R_lidar) < 0:
            R_lidar[:, 1] *= -1

        yaw_deg = np.degrees(np.arctan2(R_lidar[1, 0], R_lidar[0, 0]))
        #print(f"Vehicle Yaw (LiDAR): {yaw_deg}")
        #print(f"R_lidar Matrix:\n{R_lidar}")
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    axes.rotate(R_lidar, center=[0, 0, 0])
    #o3d.visualization.draw_geometries([point_list, bbox_o3d_vis_ego, bbox_o3d_vis, axes])
    extent = bbox.extent
    bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2])
    
    obb = o3d.geometry.OrientedBoundingBox(
        center=bbox_center_lidar,
        R=R_lidar,
        extent=bbox_extent
    )
    return obb




# 바운딩박스 저장하는 코드
def save_lidar_and_bbox(world, lidar,frame_number):
    # ✅ 2. LiDAR 위치 및 차량 bbox 저장
    label_dir = os.path.join('_out', 'label')
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, f'label_{frame_number:06d}.txt')

    actors = world.get_actors()

    with open(label_path, 'w') as f:
        # LiDAR 위치
        f.write(f"[LiDAR]\nLocation: {lidar.location.x:.3f} {lidar.location.y:.3f} {lidar.location.z:.3f}\n")
        f.write(f"Rotation: {lidar.rotation.pitch:.3f} {lidar.rotation.yaw:.3f} {lidar.rotation.roll:.3f}\n\n")

        # Ego vehicle 저장
        ego = actors.filter('vehicle.*')[0]
        ego_tf = ego.get_transform()
        ego_loc, ego_rot = ego_tf.location, ego_tf.rotation
        ego_bb = ego.bounding_box
        ego_bb_center = ego_tf.transform(ego_bb.location)
        f.write(f"[Ego Vehicle]\nLocation: {ego_loc.x:.3f} {ego_loc.y:.3f} {ego_loc.z:.3f}\n")
        f.write(f"Rotation: {ego_rot.pitch:.3f} {ego_rot.yaw:.3f} {ego_rot.roll:.3f}\n")
        f.write(f"BBox Center: {ego_bb_center.x:.3f} {ego_bb_center.y:.3f} {ego_bb_center.z:.3f}\n")
        f.write(f"BBox Extent: {ego_bb.extent.x:.3f} {ego_bb.extent.y:.3f} {ego_bb.extent.z:.3f}\n\n")

        # Other vehicles
        for actor in actors.filter('vehicle.*'):
            if actor.id == ego.id:
                continue
            tf = actor.get_transform()
            loc, rot = tf.location, tf.rotation
            bb = actor.bounding_box
            bb_center = tf.transform(bb.location)
            f.write(f"[Vehicle ID {actor.id}]\nLocation: {loc.x:.3f} {loc.y:.3f} {loc.z:.3f}\n")
            f.write(f"Rotation: {rot.pitch:.3f} {rot.yaw:.3f} {rot.roll:.3f}\n")
            f.write(f"BBox Center: {bb_center.x:.3f} {bb_center.y:.3f} {bb_center.z:.3f}\n")
            f.write(f"BBox Extent: {bb.extent.x:.3f} {bb.extent.y:.3f} {bb.extent.z:.3f}\n\n")


















# center = np.array([0.0, 0.0, 0.0])
# # 2. 크기 (extent: [width, height, depth])
# extent = np.array([2.0, 2.0, 2.0])
# # 3. 회전 행렬 (roll=0, pitch=0, yaw=0이므로 단위 행렬)
# R = np.eye(3)
# # 4. OrientedBoundingBox 생성

bbox_o3d_vis = o3d.geometry.OrientedBoundingBox()
bbox_o3d_vis.color=(0,0,1)
bbox_o3d_vis_ego = o3d.geometry.OrientedBoundingBox()
bbox_o3d_vis_ego.color=(1,0,0)



# 수정본
# def create_bbox_from_carla(bbox, vehicle_transform, lidar_transform):
#     # 1. bbox 중심 (world 좌표계 기준)
#     bbox_world_center = vehicle_transform.transform(bbox.location)
#     bbox_center_world = np.array([
#         bbox_world_center.x,
#         bbox_world_center.y,
#         bbox_world_center.z,
#         1.0
#     ])

#     # 2. LiDAR 좌표계로 변환
#     T_world_to_lidar = np.linalg.inv(np.array(lidar_transform.get_matrix()))
#     bbox_center_lidar = T_world_to_lidar @ bbox_center_world
#     bbox_center_lidar = bbox_center_lidar[:3]

#     # 3. 회전 행렬 (Open3D 기준)
#     R_vehicle = convert_carla_rotation_to_open3d(vehicle_transform.rotation)

#     # 4. 바운딩 박스 크기
#     extent = bbox.extent
#     bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2])

#     # 5. 최종 OBB
#     bbox_o3d = o3d.geometry.OrientedBoundingBox(
#         center=bbox_center_lidar,
#         R=R_vehicle,
#         extent=bbox_extent
#     )
#     bbox_o3d.color = (0, 0, 1)
#     return bbox_o3d

# def oriented_bbox_to_pointcloud(bbox: o3d.geometry.OrientedBoundingBox, step: float = 0.05) -> o3d.geometry.PointCloud:
#     """
#     Open3D의 OrientedBoundingBox 객체를 포인트 클라우드로 변환
#     - step: 선을 따라 점을 찍을 간격 (m 단위)
#     """
#     # 1. 8개의 꼭짓점
#     corners = np.asarray(bbox.get_box_points())

#     # 2. 12개의 edge 정의 (index 쌍)
#     edges = [
#         (0, 1), (1, 3), (3, 2), (2, 0),  # bottom
#         (4, 5), (5, 7), (7, 6), (6, 4),  # top
#         (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
#     ]

#     points = []

#     for start_idx, end_idx in edges:
#         p1 = corners[start_idx]
#         p2 = corners[end_idx]
#         dist = np.linalg.norm(p2 - p1)
#         num_points = max(int(dist / step), 2)
#         edge_points = np.linspace(p1, p2, num=num_points)
#         points.extend(edge_points)

#     # 3. PointCloud 객체로 변환
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(np.array(points))
#     pc.paint_uniform_color([1, 0, 0])  # 빨간색

#     return pc




def draw_lidar_axis(world, lidar_transform, length=5.0, life_time=0.1):
    """
    Ego 차량에 장착된 LiDAR의 좌표계 축 (X-red, Y-green, Z-blue)을 CARLA 월드에 그려줌.

    Args:
        world: carla.World
        lidar_transform: carla.Transform (LiDAR 센서의 world pose)
        length: 축 길이 (meter 단위)
        life_time: 시각화 지속 시간 (초)
    """
    location = lidar_transform.location
    rotation = lidar_transform.rotation

    # 회전행렬 얻기 (CARLA 좌수계)
    R_yaw = math.radians(rotation.yaw)
    R_pitch = math.radians(rotation.pitch)
    R_roll = math.radians(rotation.roll)

    cy = math.cos(R_yaw)
    sy = math.sin(R_yaw)
    cp = math.cos(R_pitch)
    sp = math.sin(R_pitch)
    cr = math.cos(R_roll)
    sr = math.sin(R_roll)

    # CARLA 좌표계 기준 회전행렬
    rotation_matrix = np.array([
        [cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])

    # 각 축 단위 벡터 (world 기준)
    x_axis = carla.Location(
        x=location.x + rotation_matrix[0, 0] * length,
        y=location.y + rotation_matrix[1, 0] * length,
        z=location.z + rotation_matrix[2, 0] * length,
    )
    y_axis = carla.Location(
        x=location.x + rotation_matrix[0, 1] * length,
        y=location.y + rotation_matrix[1, 1] * length,
        z=location.z + rotation_matrix[2, 1] * length,
    )
    z_axis = carla.Location(
        x=location.x + rotation_matrix[0, 2] * length,
        y=location.y + rotation_matrix[1, 2] * length,
        z=location.z + rotation_matrix[2, 2] * length,
    )

    # X축 (빨강)
    world.debug.draw_line(location, x_axis, thickness=0.1, color=carla.Color(255, 0, 0), life_time=life_time)
    # Y축 (초록)
    world.debug.draw_line(location, y_axis, thickness=0.1, color=carla.Color(0, 255, 0), life_time=life_time)
    # Z축 (파랑)
    world.debug.draw_line(location, z_axis, thickness=0.1, color=carla.Color(0, 0, 255), life_time=life_time)


def draw_vehicle_axes(world, vehicle, length=3.0, life_time=0.1):
    """
    차량 Local 좌표계를 world에 그려주는 코드
    X: Red (전방), Y: Green (좌측), Z: Blue (위쪽)
    """
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation

    # Transform에서 방향 벡터 추출
    forward_vector = transform.get_forward_vector()  # X축 (Red)
    right_vector = transform.get_right_vector()      # Y축 (Green)
    up_vector = transform.get_up_vector()            # Z축 (Blue)

    # 각 축 끝점 계산
    end_x = location + forward_vector * length
    end_y = location + right_vector * length
    end_z = location + up_vector * length

    # X축: 빨간색
    world.debug.draw_line(location, end_x, thickness=0.5, color=carla.Color(255, 0, 0), life_time=life_time)
    # Y축: 초록색
    world.debug.draw_line(location, end_y, thickness=0.5, color=carla.Color(0, 255, 0), life_time=life_time)
    # Z축: 파란색
    world.debug.draw_line(location, end_z, thickness=0.5, color=carla.Color(0, 0, 255), life_time=life_time)


def draw_world_axis(world, origin=carla.Location(0, 0, 0), axis_length=5.0, life_time=0.0):
    # X축 (빨강)
    world.debug.draw_arrow(
        begin=origin,
        end=origin + carla.Location(x=axis_length, y=0, z=0),
        thickness=5,
        arrow_size=5,
        color=carla.Color(1, 0, 0),  # Red
        life_time=life_time,
        persistent_lines=True
    )

    # Y축 (초록)
    world.debug.draw_arrow(
        begin=origin,
        end=origin + carla.Location(x=0, y=axis_length, z=0),
        thickness=5,
        arrow_size=5,
        color=carla.Color(0,1, 0),  # Green
        life_time=life_time,
        persistent_lines=True
    )

    # Z축 (파랑)
    world.debug.draw_arrow(
        begin=origin,
        end=origin + carla.Location(x=0, y=0, z=axis_length),
        thickness=5,
        arrow_size=5,
        color=carla.Color(0, 0,1),  # Blue
        life_time=life_time,
        persistent_lines=True
    )

# 예시 사용:
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

draw_world_axis(world, origin=carla.Location(0,0,0), axis_length=30.0, life_time=0.0)


# 수정본 디버깅을 통해 노가다...
def oriented_bbox_to_pointcloud(bbox: o3d.geometry.OrientedBoundingBox, step: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Open3D의 OrientedBoundingBox 객체를 포인트 클라우드로 변환
    - step: 선을 따라 점을 찍을 간격 (m 단위)
    - 각 꼭짓점에 번호 시각화를 포함 (디버깅용)
    """
    # 1. 8개의 꼭짓점
    corners = np.asarray(bbox.get_box_points())

    #꼭짓점 시각화용 생성
    coordinate_frames = []
    for i, pt in enumerate(corners):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.translate(pt)
        coordinate_frames.append(frame)
        print(f"Corner {i}: {pt}")  # 콘솔에도 출력
    corner_idx = [0, 1, 7, 2, 3, 6, 4, 5]
    corners = corners[corner_idx]
    # 2. 12개의 edge 정의 (index 쌍)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
    ]

    points = []
    for start_idx, end_idx in edges:
        p1 = corners[start_idx]
        p2 = corners[end_idx]
        dist = np.linalg.norm(p2 - p1)
        num_points = max(int(dist / step), 2)
        edge_points = np.linspace(p1, p2, num=num_points)
        points.extend(edge_points)

    # 3. PointCloud 객체로 변환
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    pc.paint_uniform_color([1, 0, 0])  # 빨간색

    #  꼭짓점 번호 확인용 1프레임
    #o3d.visualization.draw_geometries([pc, *coordinate_frames])

    return pc


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

VIRIDIS = np.array(matplotlib.colormaps['plasma'].colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses
def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)
    
point_list = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Carla Lidar', width=960, height=540, left=480, top=270)
vis.get_render_option().background_color = [0.05, 0.05, 0.05]
vis.get_render_option().point_size = 1
vis.get_render_option().show_coordinate_frame = True

add_open3d_axis(vis)

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, vehicle):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.vehicle=vehicle 
        self.ego=attached        
        self.frame=0
        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))
            camera_bp.set_attribute('sensor_tick', '0.0333') 

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100') 
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            #lidar.listen(self.save_lidar_image)
            lidar.listen(lambda data: self.lidar_callback(data, point_list))

            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            #lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        image.save_to_disk('_out/%08d.png' % image.frame)

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        '''       
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)
        '''
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        print(self.tics_processing)

    def lidar_callback(self, point_cloud, point_list):
        """Prepares a point cloud with intensity
        colors ready to be consumed by Open3D"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        # Isolate the 3D data
        points = data[:, :-1]

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        
        
        #points[:, :1] = -points[:, :1]

        # # An example of converting points from sensor to vehicle space if we had
        # # a carla.Transform variable named "tran":
        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        # points = np.dot(tran.get_matrix(), points.T).T
        # points = points[:, :-1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        point_cloud.save_to_disk('_out/%06d.ply' % point_cloud.frame_number)
        lidar_ego = self.world.get_actors().filter('sensor.lidar.ray_cast')[0]
        ego_transform = self.ego.get_transform()
        lidar_transform_ego = lidar_ego.get_transform()
        o3d.io.write_point_cloud('_out/pc_%06d.ply' % point_cloud.frame_number, point_list) # open3d 저장방식 사용 
        #timestamp = world.get_snapshot().timestamp.elapsed_seconds
        save_lidar_and_bbox(world,lidar_transform_ego,point_cloud.frame_number)# 바운딩박스 저장코드
        # if self.frame %2 ==1:
        #     bbox_o3d.scale(scale=0.5, center=(0,0,0))
        #     bbox_o3d.color = (0, 0, 1)  # 파란색
        #     print("self_frame",self.frame)
        #     print("bbox:",bbox_o3d)
        # if self.frame %2 ==0:
        #     bbox_o3d.scale(scale=2, center=(0,0,0))   
        #     bbox_o3d.color = (0, 0, 1)  # 파란색
        #     print("self_frame",self.frame)
        #     print("bbox:",bbox_o3d)
        # self.frame+=1
        lidars = None
        lidar_transform =None
        lidar_location = None

        ego_yaw_lidar=None # ego의 lidar좌표계에서의 각도
        yaw_lidar=None # vehicle의 lidar 좌표계에서의 각도
        ego_transform = self.ego.get_transform()
        ego_yaw_deg = ego_transform.rotation.yaw
        ego_yaw_rad = np.radians(ego_yaw_deg)
        
        # ✅ Ego의 반대 회전 행렬 (좌표계 회전 효과 상쇄용)
        #ego_inverse_R = R.from_euler('z', -ego_yaw_rad).as_matrix()
        for vehicle in self.world.get_actors().filter('vehicle.*'):
                draw_vehicle_axes(world, vehicle)   
                if vehicle.id == self.ego.id: # 이게 원본 이거 해제 하면됨
                    vehicle_ego = vehicle
                    
                else:
                    vehicle_veh = vehicle
                    
                    
                
                
                lidar_location_ego = lidar_transform_ego.location
                lidar_rotation_ego = lidar_transform_ego.rotation
                #draw_lidar_axis(world, lidar_transform_ego)

                #다른 차량 라이다 정보
                lidar_veh = self.world.get_actors().filter('sensor.lidar.ray_cast_semantic')[0]
                lidar_transform_veh = lidar_veh.get_transform()
                #lidar_location_veh = lidar_transform_veh.location

                #print(f"LiDAR 센서 중심 좌표 (월드 기준): x={lidar_location_ego.x:.2f}, y={lidar_location_ego.y:.2f}, z={lidar_location_ego.z:.2f}")
                print(f"LiDAR 센서  Location: {lidar_location_ego}, Rotation: {lidar_rotation_ego}")
                #print(f"중심 회전 (월드 기준): Rotation{lidar_location_ego.rotation}")
                bbox = vehicle.bounding_box
                
                transform = vehicle.get_transform()
                # 바운딩박스의 위치를 월드 기준으로 변환
                bbox_world = carla.BoundingBox(
                    transform.transform(bbox.location),  # location: world 기준
                    bbox.extent# 크기 그대로
                )
                rotation = transform.rotation  # 회전만 따로 추출
                #print(f" 환산 전 ID: {vehicle.id}, Type: {vehicle.type_id}, Location: {vehicle.get_location()},bbox:{bbox_world},Rotation:{rotation}")
                #print(f"Rotation:{rotation}")
                # 바운딩박스 시각화
                self.world.debug.draw_box(
                    bbox_world,
                    rotation,
                    thickness=0.1,
                    color=carla.Color(255, 0, 0),
                    life_time=0.1,
                    persistent_lines=True
                )                               
                #bbox = vehicle.bounding_box
                vehicle_transform = vehicle.get_transform()
                #lidar_transform = lidars.get_transform()
                if vehicle.id == self.ego.id:
                    # Ego 차량 (자기 자신 기준 LiDAR)
                    vehicle_transform = self.ego.get_transform()  #ego 차량 transform 확실히 지정
                    bbox = vehicle.bounding_box
                    #print("==================================")
                    #print(f"[Vehicle ID] {vehicle.id}")
                    print(f"[ego_Vehicle WORLD] Location: {vehicle_transform.location}, Rotation: {vehicle_transform.rotation}")
                    print(f"[ego_BBox]{bbox}")
                    #bbox_o3d_ego = create_bbox_from_carla(bbox, vehicle_transform, lidar_transform_ego)
                    bbox_o3d_ego = create_bbox_from_actor(vehicle, lidar_transform_ego,ego_transform,is_ego=True)
                    bbox_o3d_ego.color = (0, 1, 0)
                    bbox_o3d_vis_ego.center = bbox_o3d_ego.center
                    bbox_o3d_vis_ego.R = bbox_o3d_ego.R
                    bbox_o3d_vis_ego.extent = bbox_o3d_ego.extent
                    #print(f"[ego_BBox LIDAR] Center: {bbox_o3d_ego.center}")
                    #print(f"[ego_BBox LIDAR] Rotation Matrix:\n{bbox_o3d_ego.R}")
                    #ego_yaw_lidar = extract_yaw_from_open3d_R(bbox_o3d_ego.R)
                    #print(f"[ego_BBox LIDAR] Yaw(deg): {ego_yaw_lidar:.2f}")
                    #print("==================================")
                    # print()
                    # print(f"bbox_R값{bbox_o3d_vis_ego.R}")
                    # print("ego 차량의 r 값",extract_yaw_from_open3d_R(bbox_o3d_vis_ego.R))

                else: # 이게 원본임
                    vehicle_transform = vehicle_veh.get_transform()
                    bbox = vehicle.bounding_box
                    
                    # if abs(ego_yaw_deg-vehicle_veh.get_transform().rotation.yaw) > 5:  # ✅ 아주 미세한 경우 무시하고, 일정 각도 이상일 때만
                    #     ego_yaw_deg=abs(ego_yaw_deg-vehicle_veh.get_transform().rotation.yaw)
                    #     ego_inverse_R = R.from_euler('z', ego_yaw_rad).as_matrix()
                    # else:
                    #     ego_inverse_R = np.eye(3)  # 단위행렬 (회전 없음)
                    #print("==================================")
                    #print(f"[Vehicle ID] {vehicle.id}")
                    print(f"[Vehicle WORLD] Location: {vehicle_transform.location}, Rotation: {vehicle_transform.rotation}")
                    print(f"[Vehicle_BBox]{bbox}")
                    #bbox_o3d = create_bbox_from_carla(bbox, vehicle_transform, lidar_transform_ego)
                    bbox_o3d = create_bbox_from_actor(vehicle, lidar_transform_ego,ego_transform,is_ego=False)
                    #bbox_o3d.rotate(ego_inverse_R, center=(bbox_o3d.center))
                    bbox_o3d.color = (0, 0, 1)  # 파란색
                    bbox_o3d_vis.center = bbox_o3d.center
                    bbox_o3d_vis.R = bbox_o3d.R
                    bbox_o3d_vis.extent = bbox_o3d.extent
                    #print(f"[BBox LIDAR] Center: {bbox_o3d.center}")
                    #print(f"[BBox LIDAR] Rotation Matrix:\n{bbox_o3d.R}")
                    #yaw_lidar = extract_yaw_from_open3d_R(bbox_o3d.R)
                    #print(f"[BBox LIDAR] Yaw(deg): {yaw_lidar:.2f}")
                    print("==================================")
                    # print()
                    # print(f"bbox_R값{bbox_o3d_vis.R}")
                    # print("다른 차량의 r 값",extract_yaw_from_open3d_R(bbox_o3d_vis.R))
                #bbox_o3d = create_bbox_from_carla(bbox, vehicle_transform, lidar_transform)                
                #print(f" 환산 후 ID: {vehicle.id}, Type: {vehicle.type_id}, Location: {vehicle.get_location()},bbox:{bbox_world}")
                #bbox_o3d_vis = bbox_o3d
                

                #if ego_yaw_lidar-yaw_lidar < 1:
                    
                #print("변환후 R값:",bbox_o3d_vis.R)
                #bbox_pc = oriented_bbox_to_pointcloud(bbox_o3d, step=0.1)   
                #o3d.io.write_point_cloud('_out/bbox_%06d.ply' % point_cloud.frame_number,bbox_pc)
                #print(f"ego의 Rotation{lidar_transform_ego.get_transform().rotation}") 
        #gap= (vehicle_ego.get_transform().rotation.yaw-vehicle_veh.get_transform().rotation.yaw) + 360
        #print("두 차량의 차이:",vehicle_ego.get_transform().rotation, vehicle_veh.get_transform().rotation,gap )        
        #R = bbox_o3d_vis.get_rotation_matrix_from_xyz((0, 0,(gap/180*np.pi)))
        #bbox_o3d_vis.rotate(R, center=bbox_o3d_vis.get_center())
        
    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """
    
    
    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:
        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)  # 모든 신호등 초록불로 설정
            traffic_light.set_green_time(9999.0)                   # 초록불 유지 시간 길게 설정
            traffic_light.freeze(True)                             # 상태 변경 방지
        

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Instanciating the vehicle to which we attached the sensors
        #bp = random.choice(world.get_blueprint_library().filter(args.filter))
        bp= world.get_blueprint_library().find("vehicle.mini.cooper")
        spawn_points = world.get_map().get_spawn_points()
        # 기존 spawn_point에서 복사
        spawn_tf = spawn_points[122]
        spawn_ttf= spawn_points[26]
        # 원하는 방향으로 회전 설정 (예: yaw = 90도)
        spawn_tf.rotation.yaw = 45.0  # 동쪽 방향
        spawn_ttf.rotation.yaw = 45.0
        ego = world.try_spawn_actor(bp,spawn_ttf )
        vehicle_list.append(ego)
        vehicle = world.spawn_actor(bp, spawn_tf)
        vehicle_list.append(vehicle)
        #vehicle_list.append(vehicle)
        #world.try_spawn_actor(bp,(spawn_points[0]))
        #spawn_points = world.get_map().get_spawn_points()
        #world.try_spawn_actor(bp,(spawn_points[93]))
        ego.set_autopilot(True)
        vehicle.set_autopilot(True)
        


        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1,1], window_size=[args.width, args.height])
        
        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        #SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)), 
        #              vehicle, {}, display_pos=[0, 0])
        
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=+00)), ego, {}, display_pos=[0, 0],vehicle=vehicle)
        
        #SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)), 
        #              vehicle, {}, display_pos=[0, 2])
        #SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)), 
        #              vehicle, {}, display_pos=[1, 1])

        SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0,y=0,z=2.4),carla.Rotation(yaw=0.0,roll=0.0,pitch=0.0)), 
                      ego, {'channels' : '64', 'range' : '100',  'points_per_second': '1300000', 'rotation_frequency': '20','upper_fov': '2.0','lower_fov': '-24.8'}, display_pos=[0, 1],vehicle=vehicle)
        SensorManager(world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0, y=0,z=2.4), carla.Rotation(yaw=0.0,roll=0.0,pitch=0.0)),
                     vehicle, {'channels' : '64', 'range' : '100', 'points_per_second': '1300000', 'rotation_frequency': '20'}, display_pos=[0,0],vehicle=vehicle) #기존[1,2]

        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()

        # # create open3d window
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name='Carla Lidar', width=960, height=540, left=480, top=270)
        # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        # vis.get_render_option().point_size = 1
        # vis.get_render_option().show_coordinate_frame = True
        #add_open3d_axis(vis)
        #point_list = o3d.geometry.PointCloud()
        #lidar.listen(lambda data: lidar_callback(data, point_list))
        frame = 0
        
        while True:
            # Carla Tick
            world.tick()
            # Render received data
            display_manager.render()
            
            
            # update 3d lidar data
            #print("bbox:",bbox_o3d_vis)
            #print("bbox_ego",bbox_o3d_vis_ego)
            if frame == 2:
                vis.add_geometry(point_list)
                vis.add_geometry(bbox_o3d_vis)  # Open3D 시각화 창에 추가
                vis.add_geometry(bbox_o3d_vis_ego) 
                
            vis.update_geometry(point_list)
            vis.update_geometry(bbox_o3d_vis)
            vis.update_geometry(bbox_o3d_vis_ego)

            vis.poll_events()
            vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                    #if display_manager:
                    #    display_manager.destroy()

                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        
                        #if display_manager:
                        #    display_manager.destroy()

                        break

            if call_exit:
                    print("call_exit")
                #if original_settings:
                    #world.apply_settings(original_settings)

                    # if (world and world.recording_enabled):
                    #     client.stop_recorder()

                    # if world is not None:
                    #     world.destroy()

                    #pygame.quit()
            
                    # if display_manager:
                    #     display_manager.destroy()
                        
                    break
            frame += 1

    finally:
        print("finally")
        world.apply_settings(original_settings)
        if display_manager:
            display_manager.destroy()
        # 모든 차량 객체 가져오기
        vehicles = world.get_actors().filter('vehicle.*')[0]
       # 차량 제거 (0.10.0 기준: 개별 destroy 호출)
        for actor in vehicle_list:
            if actor.is_alive:
                print(f"Destroying: {actor.type_id}")
                actor.destroy()
        # 제거
        # world.destroy_actors(vehicles)
        # print(f"{len(vehicles)} vehicles destroyed!")

        #client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        #[carla.command.DestroyActor(x) for x in vehicle_list]
        # if world is not None:
        #     world.destroy()
        
        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()