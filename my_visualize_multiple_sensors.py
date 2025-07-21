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
from scipy.spatial.transform import Rotation as R
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



def convert_carla_rotation_to_open3d(rotation: carla.Rotation) -> np.ndarray:
    """
    CARLA 좌수좌표계의 roll, pitch, yaw를 Open3D 우수좌표계 기준의 회전 행렬로 변환
    ZYX (yaw → pitch → roll) 순서로 회전
    """
    yaw = -np.radians(rotation.yaw)
    pitch = -np.radians(rotation.pitch)
    roll = np.radians(rotation.roll)
    rot = R.from_euler('ZYX', [yaw, pitch, roll])
    return rot.as_matrix()  # (3,3)

def create_bbox_from_carla(bbox: carla.BoundingBox,
                            vehicle_transform: carla.Transform,
                            lidar_transform: carla.Transform) -> o3d.geometry.OrientedBoundingBox:
    """
    CARLA vehicle의 BoundingBox를 LiDAR 좌표계 기준 Open3D OBB로 변환
    """
    # 1. 바운딩 박스 중심 (월드 기준)
    bbox_world_center = vehicle_transform.transform(bbox.location)
    bbox_center_world = np.array([
        bbox_world_center.x,
        bbox_world_center.y,
        bbox_world_center.z
    ])

    # 2. 차량 회전 행렬 (Open3D 우수 기준)
    R_vehicle = convert_carla_rotation_to_open3d(vehicle_transform.rotation)  # (3x3)

    # 3. LiDAR의 월드 기준 변환행렬
    T_world_to_lidar = np.linalg.inv(np.array(lidar_transform.get_matrix()))  # (4x4)

    # 4. bbox 중심 → lidar 기준으로 변환
    bbox_center_world_h = np.append(bbox_center_world, 1.0)  # (4,)
    bbox_center_lidar_h = T_world_to_lidar @ bbox_center_world_h
    bbox_center_lidar = bbox_center_lidar_h[:3]

    # 5. 차량 회전 → lidar 좌표계로 변환 (회전만 적용)
    R_lidar = T_world_to_lidar[:3, :3] @ R_vehicle

    # 6. 좌표계 반전 보정 (CARLA 좌수 → Open3D 우수)
    bbox_center_lidar[0] *= -1          # X 좌표 반전
    R_lidar[:, 0] *= -1                 # X축 방향 반전

    # 7. 바운딩 박스 크기 (extent는 반길이 → full size)
    extent = bbox.extent
    bbox_extent = np.array([extent.x * 2, extent.y * 2, extent.z * 2])

    # 8. Open3D OBB 생성
    obb = o3d.geometry.OrientedBoundingBox(center=bbox_center_lidar,
                                           R=R_lidar,
                                           extent=bbox_extent)
    obb.color = (0, 0, 1)  # 파란색
    return obb


# center = np.array([0.0, 0.0, 0.0])
# # 2. 크기 (extent: [width, height, depth])
# extent = np.array([2.0, 2.0, 2.0])
# # 3. 회전 행렬 (roll=0, pitch=0, yaw=0이므로 단위 행렬)
# R = np.eye(3)
# # 4. OrientedBoundingBox 생성

bbox_o3d_vis = o3d.geometry.OrientedBoundingBox()
bbox_o3d_vis.color=(0,0,1)


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
        points[:, :1] = -points[:, :1]

        # # An example of converting points from sensor to vehicle space if we had
        # # a carla.Transform variable named "tran":
        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        # points = np.dot(tran.get_matrix(), points.T).T
        # points = points[:, :-1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        #point_cloud.save_to_disk('_out/%06d.ply' % point_cloud.frame_number)
        
        #o3d.io.write_point_cloud('_out/pc_%06d.ply' % point_cloud.frame_number, point_list) # open3d 저장방식 사용 
        
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


        for vehicle in self.world.get_actors().filter('vehicle.*'):
                   
            if vehicle.id != self.ego.id:
                lidar_ego = self.world.get_actors().filter('sensor.lidar.ray_cast')[0]
                lidar_transform_ego = lidar_ego.get_transform()
                lidar_location_ego = lidar_transform_ego.location

                lidar_veh = self.world.get_actors().filter('sensor.lidar.ray_cast_semantic')[0]
                lidar_transform_veh = lidar_veh.get_transform()
                lidar_location_veh = lidar_transform_veh.location

                print(f"LiDAR 센서 중심 좌표 (월드 기준): x={lidar_location_ego.x:.2f}, y={lidar_location_ego.y:.2f}, z={lidar_location_ego.z:.2f}")

                bbox = vehicle.bounding_box
                transform = vehicle.get_transform()
                # 바운딩박스의 위치를 월드 기준으로 변환
                bbox_world = carla.BoundingBox(
                    transform.transform(bbox.location),  # location: world 기준
                    bbox.extent# 크기 그대로
                )
                rotation = transform.rotation  # 회전만 따로 추출
                print(f" 환산 전 ID: {vehicle.id}, Type: {vehicle.type_id}, Location: {vehicle.get_location()},bbox:{bbox_world},Rotation:{rotation}")
                # 바운딩박스 시각화
                self.world.debug.draw_box(
                    bbox_world,
                    rotation,
                    thickness=0.1,
                    color=carla.Color(255, 0, 0),
                    life_time=0.1,
                    persistent_lines=True
                )                               
                
                bbox = vehicle.bounding_box
                vehicle_transform = vehicle.get_transform()
                ego_transform = self.ego.get_transform()
                #lidar_transform = lidars.get_transform()
                
                #bbox_o3d = create_bbox_from_carla(bbox, vehicle_transform, lidar_transform)
                bbox_o3d = create_bbox_from_carla(bbox, vehicle_transform, lidar_transform_ego)
                bbox_o3d.color = (0, 0, 1)  # 파란색
                #bbox_o3d_vis = bbox_o3d
                bbox_o3d_vis.center = bbox_o3d.center
                bbox_o3d_vis.R = bbox_o3d.R
                bbox_o3d_vis.extent = bbox_o3d.extent
                #print(f" 환산 후 ID: {vehicle.id}, Type: {vehicle.type_id}, Location: {vehicle.get_location()},bbox:{bbox_world}")      
                print("변환후 R값:",bbox_o3d_vis.R)
                #bbox_pc = oriented_bbox_to_pointcloud(bbox_o3d, step=0.1)
                #o3d.io.write_point_cloud('_out/bbox_%06d.ply' % point_cloud.frame_number,bbox_pc)
            

            
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
        #spawn_ttf= spawn_points[26]
        # 원하는 방향으로 회전 설정 (예: yaw = 90도)
        spawn_tf.rotation.yaw = 45.0  # 동쪽 방향
        #spawn_ttf.rotation.yaw = 45.0
        ego = world.try_spawn_actor(bp,spawn_points[26] )
        vehicle_list.append(ego)
        vehicle = world.spawn_actor(bp, spawn_points[122])
        vehicle_list.append(vehicle)
        #vehicle_list.append(vehicle)
        #world.try_spawn_actor(bp,(spawn_points[0]))
        #spawn_points = world.get_map().get_spawn_points()
        #world.try_spawn_actor(bp,(spawn_points[93]))
        ego.set_autopilot(True)
        #vehicle.set_autopilot(True)
        


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
            print("bbox:",bbox_o3d_vis)
            if frame == 2:
                vis.add_geometry(point_list)
                vis.add_geometry(bbox_o3d_vis)  # Open3D 시각화 창에 추가
                
            vis.update_geometry(point_list)
            vis.update_geometry(bbox_o3d_vis)

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