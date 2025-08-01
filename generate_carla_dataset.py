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
import msvcrt
import cv2


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


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class SensorManager:
    def __init__(self, world, sensor_type, transform, attached, sensor_options, vehicle):
        self.surface = None
        self.world = world
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.vehicle=vehicle 
        self.ego=attached        
        self.frame=0
        self.time_processing = 0.0
        self.tics_processing = 0
        self.point_list = o3d.geometry.PointCloud()
        self.lidar_origin=[0,0,0]
        self.ego_origin=None
        self.ego_yaw=0.0
        self.lidar_yaw=0.0
        self.vehicle_center=[1,1,1]
        self.vehicle_extent=[2,2,2]
        self.vehicle_yaw=0
        self.ego_center=[1,1,1]
        self.ego_extent=[2,2,2]
        self.last_camera_image = None
        
    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            #disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(1024))
            camera_bp.set_attribute('image_size_y', str(720))
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
            lidar.listen(lambda data: self.lidar_callback(data, self.point_list))

            return lidar        
               
        else:
            return None

    def get_ego_center(self):
        return self.ego_center
    
    def get_ego_extent(self):
        return self.ego_extent
    
    def get_vehicle_center(self):
        return self.vehicle_center
    
    def get_vehicle_extent(self):
        return self.vehicle_extent

    def get_vehicle_yaw(self):
        return self.vehicle_yaw
    
    def get_sensor(self):
        return self.sensor
    
    def get_ego_yaw(self):
        return self.ego_yaw
    
    def get_lidar_yaw(self):
        return self.lidar_yaw

    def get_lidar_origin(self):
        return self.lidar_origin

    def get_ego_origin(self):
        return self.ego_origin
    
    def get_point(self):
        return self.point_list

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        self.last_camera_image=array.copy()
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
        print(f"lidar_callback{self.timer.time()}")

        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        # Isolate the 3D data
        points = data[:, :-1]
        points[:, 1] = -points[:, 1]

        # # An example of converting points from sensor to vehicle space if we had
        # # a carla.Transform variable named "tran":
        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        # points = np.dot(tran.get_matrix(), points.T).T
        
        #points = points[:, :-1]
        colors=np.ones_like(points)
        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(np.float64(colors))
        point_cloud.save_to_disk('_out/%06d.ply' % point_cloud.frame_number)#pointcloud 저장
        lidar_ego = self.world.get_actors().filter('sensor.lidar.ray_cast')[0]
        #ego_transform = self.ego.get_transform()
        #lidar_transform_ego = lidar_ego.get_transform()
        lidar_tf = lidar_ego.get_transform()
        self.lidar_origin = np.array([lidar_tf.location.x, lidar_tf.location.y, lidar_tf.location.z])
        self.lidar_origin[1]*=-1
        self.lidar_yaw = lidar_tf.rotation.yaw
        ego_transform = self.ego.get_transform()
        lidar_transform_ego = lidar_ego.get_transform()
        #o3d.io.write_point_cloud('_out/pc_%06d.ply' % point_cloud.frame_number, point_list) # open3d 저장방식 사용 
        #timestamp = world.get_snapshot().timestamp.elapsed_seconds
        save_lidar_and_bbox(self.world,lidar_transform_ego,point_cloud.frame_number)# 바운딩박스 저장코드
        #points_trans = points + self.lidar_origin
        #points_rot = (points_trans - lidar_origin) @ lidar_R.T + lidar_origin
        #point_list.points = o3d.utility.Vector3dVector(points_rot)
        #point_list.colors = o3d.utility.Vector3dVector(int_color)
        
        #point_list.translate(self.lidar_origin)
        #point_list.rotate(lidar_R, center=self.lidar_origin)
        
        #lidar_axis.translate(lidar_origin)
        #lidar_axis.rotate(lidar_R, center=lidar_origin)

        # --- 4. Ego Axis ---
        ego_tf = self.ego.get_transform()
        self.ego_origin = np.array([ego_tf.location.x, ego_tf.location.y, ego_tf.location.z])
        self.ego_origin[1]*=-1
        self.ego_yaw = ego_tf.rotation.yaw
        
        #self.ego_axis.translate(ego_origin)
        #self.ego_axis.rotate(ego_R, center=ego_origin)
        
        
        
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            tf = vehicle.get_transform()
            bb = vehicle.bounding_box
            center = tf.transform(bb.location)
            center_np = np.array([center.x, center.y, center.z])
            center_np[1] *= -1
            extent = bb.extent
            yaw = -np.deg2rad(tf.rotation.yaw)
            #R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
            if vehicle.id == self.ego.id:
                    #Ego 차량 (자기 자신 기준 LiDAR)
                    # tf = self.ego.get_transform()
                    # bb = self.ego.bounding_box
                    # center = tf.transform(bb.location)
                    # extent = bb.extent
                    # yaw = -np.deg2rad(tf.rotation.yaw)
                    self.ego_center=center_np
                    print(self.ego_center)
                    self.ego_extent=np.array([extent.x*2, extent.y*2, extent.z*2])
                    #R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
                    

            else: # 이게 원본임
                #vehicle_transform = vehicle_veh.get_transform()
                # bbox = vehicle.bounding_box
                # tf = vehicle.get_transform()
                # center = tf.transform(bbox.location)
                # extent = bbox.extent
                # yaw = -np.deg2rad(tf.rotation.yaw)
                self.vehicle_center=center_np
                self.vehicle_extent=np.array([extent.x*2, extent.y*2, extent.z*2])
                self.vehicle_yaw=yaw
                #R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])

                
        """
        Prepares a point cloud with intensity
        colors ready to be consumed by Open3D"""
        """
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        # Isolate the intensity and compute a color for it
        
        
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
        point_list.colors = o3d.utility.Vector3dVector(np.ones_like(points))
        point_cloud.save_to_disk('_out/%06d.ply' % point_cloud.frame_number)
        lidar_ego = self.world.get_actors().filter('sensor.lidar.ray_cast')[0]

        #lidar_location=lidar_ego.get_transform().location
        #lidar_origin=[lidar_location.x,lidar_location.y,lidar_location.z]
        # print(f"lidar_origin 값:{lidar_origin}")
        #lidar_R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi * -lidar_ego.get_transform().rotation.yaw / 180))

       
        ego_transform = self.ego.get_transform()
        lidar_transform_ego = lidar_ego.get_transform()
        o3d.io.write_point_cloud('_out/pc_%06d.ply' % point_cloud.frame_number, point_list) # open3d 저장방식 사용 
        #timestamp = world.get_snapshot().timestamp.elapsed_seconds
        #save_lidar_and_bbox(world,lidar_transform_ego,point_cloud.frame_number)# 바운딩박스 저장코드
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
        

        #ego_yaw_lidar=None # ego의 lidar좌표계에서의 각도
        #yaw_lidar=None # vehicle의 lidar 좌표계에서의 각도
        #ego_transform = self.ego.get_transform()
        #ego_yaw_deg = ego_transform.rotation.yaw
        #ego_yaw_rad = np.radians(ego_yaw_deg)
        
        #c추기
        
        lidar_tf = lidar_ego.get_transform()
        lidar_origin = np.array([lidar_tf.location.x, lidar_tf.location.y, lidar_tf.location.z])
        lidar_origin[1]*=-1
        lidar_yaw = lidar_tf.rotation.yaw
        lidar_R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi * -lidar_yaw / 180))
        points_trans = points + lidar_origin
        #points_rot = (points_trans - lidar_origin) @ lidar_R.T + lidar_origin
        #point_list.points = o3d.utility.Vector3dVector(points_rot)
        #point_list.colors = o3d.utility.Vector3dVector(int_color)
        point_list.translate(lidar_origin)
        point_list.rotate(lidar_R, center=lidar_origin)
        #lidar_axis.translate(lidar_origin)
        #lidar_axis.rotate(lidar_R, center=lidar_origin)

        # --- 4. Ego Axis ---
        ego_tf = self.ego.get_transform()
        ego_origin = np.array([ego_tf.location.x, ego_tf.location.y, ego_tf.location.z])
        ego_origin[1]*=-1
        ego_yaw = ego_tf.rotation.yaw
        ego_R = o3d.geometry.get_rotation_matrix_from_xyz((0,0,np.pi * -ego_yaw / 180))
        #ego_axis.translate(ego_origin)
        #ego_axis.rotate(ego_R, center=ego_origin)
        
        lidar_ax=o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=lidar_origin)
        lidar_ax.rotate(lidar_R, center=lidar_origin)
        self.display_man.set_lidar_axis(lidar_ax)
        #self.lidar_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=lidar_origin)
        for vehicle in self.world.get_actors().filter('vehicle.*'):
               #draw_vehicle_axes(world, vehicle)   


                # if vehicle.id == self.ego.id: # 이게 원본 이거 해제 하면됨
                #     #vehicle_ego = vehicle
                # else:
                #     vehicle_veh = vehicle 
                #lidar_location_ego = lidar_transform_ego.location
                #lidar_rotation_ego = lidar_transform_ego.rotation
                #draw_lidar_axis(world, lidar_transform_ego)


                tf = vehicle.get_transform()
                bb = vehicle.bounding_box
                center = tf.transform(bb.location)
                center_np = np.array([center.x, center.y, center.z])
                center_np[1] *= -1
                extent = bb.extent
                yaw = -np.deg2rad(tf.rotation.yaw)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
                #center_np = np.array([center.x, -center.y, center.z])
                # BBox 회전 적용 (원점기준 이동+회전)
                #center_np = center_np + lidar_origin
                #center_rot = (center_np - lidar_origin) @ lidar_R.T + lidar_origin
                #R_total = lidar_R @ R
                #다른 차량 라이다 정보
                lidar_veh = self.world.get_actors().filter('sensor.lidar.ray_cast_semantic')[0]
                lidar_transform_veh = lidar_veh.get_transform()
                #lidar_location_veh = lidar_transform_veh.location

                #print(f"LiDAR 센서 중심 좌표 (월드 기준): x={lidar_location_ego.x:.2f}, y={lidar_location_ego.y:.2f}, z={lidar_location_ego.z:.2f}")
                # print(f"LiDAR 센서  Location: {lidar_location_ego}, Rotation: {lidar_rotation_ego}")
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
                    thickness=0.05,
                    color=carla.Color(255,0,0),
                    life_time=0.1,
                    persistent_lines=True
                )                               
                #bbox = vehicle.bounding_box
                vehicle_transform = vehicle.get_transform()
                #lidar_transform = lidars.get_transform()
                if vehicle.id == self.ego.id:
                    # Ego 차량 (자기 자신 기준 LiDAR)
                    # tf = self.ego.get_transform()
                    # bb = self.ego.bounding_box
                    # center = tf.transform(bb.location)
                    # extent = bb.extent
                    # yaw = -np.deg2rad(tf.rotation.yaw)
                    # R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
                    bbox_o3d_vis_ego.center = center_np
                    bbox_o3d_vis_ego.extent = np.array([extent.x*2, extent.y*2, extent.z*2])
                    bbox_o3d_vis_ego.R = R
                    bbox_o3d_vis_ego.color = [0,1,0] # green

                else: # 이게 원본임
                    #vehicle_transform = vehicle_veh.get_transform()
                    # bbox = vehicle.bounding_box
                    # tf = vehicle.get_transform()
                    # center = tf.transform(bbox.location)
                    # extent = bbox.extent
                    # yaw = -np.deg2rad(tf.rotation.yaw)
                    # R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
                    bbox_o3d_vis.center = center_np
                    bbox_o3d_vis.extent = np.array([extent.x*2, extent.y*2, extent.z*2])
                    bbox_o3d_vis.R = R
                    bbox_o3d_vis.color = [1,0,0] # red
                    
                    
                    # bbox_o3d.color = (0, 0, 1)  # 파란색
                    # bbox_o3d_vis.center = bbox_o3d.center
                    # bbox_o3d_vis.R = bbox_o3d.R
                    # bbox_o3d_vis.extent = bbox_o3d.extent
                    #print(f"[BBox LIDAR] Center: {bbox_o3d.center}")
                    #print(f"[BBox LIDAR] Rotation Matrix:\n{bbox_o3d.R}")
                    #yaw_lidar = extract_yaw_from_open3d_R(bbox_o3d.R)
                    #print(f"[BBox LIDAR] Yaw(deg): {yaw_lidar:.2f}")
                    # print("==================================")
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
    """    
    def destroy(self):
        self.sensor.destroy()
        self.vis.destroy_window()



def run_simulation(args, client):
    import numpy as np
    import time

    vehicle_list = []
    vis = None

    try:
        # CARLA 서버 초기화
        world = client.get_world()
        original_settings = world.get_settings()
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(9999.0)
            traffic_light.freeze(True)
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 차량 생성 설정
        bp = world.get_blueprint_library().find("vehicle.mini.cooper")
        spawn_points = world.get_map().get_spawn_points()
        spawn_tf = spawn_points[122]
        spawn_ttf = spawn_points[26]
        spawn_tf.rotation.yaw = 45.0
        spawn_ttf.rotation.yaw = 45.0
        ego = world.try_spawn_actor(bp, spawn_ttf)
        vehicle_list.append(ego)
        vehicle = world.spawn_actor(bp, spawn_tf)
        vehicle_list.append(vehicle)
        ego.set_autopilot(True)
        vehicle.set_autopilot(True)

        # 센서 초기화
        lidar = SensorManager(
            world, 'LiDAR',
            carla.Transform(carla.Location(x=0, y=0, z=2.4),
                            carla.Rotation(yaw=0.0, roll=0.0, pitch=0.0)),
            ego, {'channels': '64', 'range': '100', 'points_per_second': '1300000',
                  'rotation_frequency': '20', 'upper_fov': '2.0', 'lower_fov': '-24.8'},
            vehicle=vehicle
        )
        camera= SensorManager(world, 'RGBCamera', carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=+00)),ego,{ },vehicle=vehicle)
        # Open3D Window, Geometry 초기화 및 생성
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='Carla Lidar', width=1024, height=720)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = False

        # 최초 1회 geometry add (원본 점군은 numpy로 보존)
        base_pc = lidar.get_point()  
        raw_points = np.asarray(base_pc.points).copy()
        raw_colors = np.asarray(base_pc.colors).copy()

        # dummy geometry 객체 (렌더링용)
        pc_vis = o3d.geometry.PointCloud()
        vis.add_geometry(pc_vis)

        bbox_o3d_vis = o3d.geometry.OrientedBoundingBox()
        bbox_o3d_vis.color = (1, 0, 0)
        vis.add_geometry(bbox_o3d_vis)

        bbox_o3d_vis_ego = o3d.geometry.OrientedBoundingBox()
        bbox_o3d_vis_ego.color = (0, 1, 0)
        vis.add_geometry(bbox_o3d_vis_ego)

        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        vis.add_geometry(world_axis)

        # lidar_axis는 별도 관리 (매프레임 remove/add)
        lidar_origin = lidar.get_lidar_origin()
        lidar_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=lidar_origin)
        vis.add_geometry(lidar_axis)

        
        vis.get_view_control().set_zoom(0.3)
        vis.get_view_control().set_front([0, 0, 1])
        vis.get_view_control().set_up([0, 1, 0])
        vis.get_view_control().set_lookat(lidar_origin)

        
        should_exit = {'flag': False}
        def esc_callback(vis):
            should_exit['flag'] = True
            return False
        vis.register_key_callback(256, esc_callback)

        frame = 0
        while not should_exit['flag']:
            world.tick()
            new_points = lidar.get_point()
            raw_points = np.asarray(new_points.points).copy()
            raw_colors = np.asarray(new_points.colors).copy()

            pc_vis.points = o3d.utility.Vector3dVector(raw_points)
            pc_vis.colors = o3d.utility.Vector3dVector(raw_colors)

            lidar_R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -np.pi * lidar.get_lidar_yaw() / 180))
            lidar_origin = lidar.get_lidar_origin()

            # 여기서만 변환을 적용(pc_vis는 매 프레임 원본으로 갱신됨)
            pc_vis.translate(lidar_origin)
            pc_vis.rotate(lidar_R, center=lidar_origin)

            # BBox 갱신 (geometry 누적 변환X)
            bbox_o3d_vis_ego.center = lidar.get_ego_center()
            bbox_o3d_vis_ego.extent = lidar.get_ego_extent()
            bbox_o3d_vis_ego.R = lidar_R
            bbox_o3d_vis_ego.color = [0, 1, 0]

            bbox_o3d_vis.center = lidar.get_vehicle_center()
            bbox_o3d_vis.extent = lidar.get_vehicle_extent()
            bbox_o3d_vis.R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, lidar.get_vehicle_yaw()))
            bbox_o3d_vis.color = [1, 0, 0]

            # Axis 갱신 (remove -> 새로 add)
            vis.remove_geometry(lidar_axis, reset_bounding_box=False)
            lidar_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=lidar_origin)
            lidar_axis.rotate(lidar_R, center=lidar_origin)
            vis.add_geometry(lidar_axis)
            vis.update_geometry(pc_vis)
            vis.update_geometry(bbox_o3d_vis)
            vis.update_geometry(bbox_o3d_vis_ego)
            vis.update_geometry(world_axis)
            vis.get_view_control().set_lookat(lidar_origin)
            if camera.last_camera_image is not None:
                cv2.imshow("Front Camera", camera.last_camera_image)
                if cv2.waitKey(1) == 27:  # ESC 누르면
                    should_exit['flag'] = True
            # geometry 업데이트
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)
            frame += 1

    finally:
        print("finally")
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()
        world.apply_settings(original_settings)
        for actor in vehicle_list:
            if actor.is_alive:
                print(f"Destroying: {actor.type_id}")
                actor.destroy()


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