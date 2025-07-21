# import open3d as o3d
# import numpy as np
# import os

# # 포인트클라우드 폴더 경로
# lidar_dir = r'_out/pc_222789.ply'    # 혹은 상위 폴더
# label_dir =r'_out/label_222789.txt'
# pcd=o3d.io.read_point_cloud(lidar_dir)

# filepath = os.path.join(lidar_dir, filename)
# print(f'시각화 중: {filename}')
# pcd = o3d.io.read_point_cloud(filepath)
# # --- 포인트 색상을 모두 흰색으로 지정 ---
# num_points = np.asarray(pcd.points).shape[0]
# white_colors = np.ones((num_points, 3))  # (N,3), 값은 1.0(흰색)
# pcd.colors = o3d.utility.Vector3dVector(white_colors)
# points = np.asarray(pcd.points)
# #points[:, :1] = -points[:, :1]
# pcd.points = o3d.utility.Vector3dVector(points)
# # --- 시각화: 흰색 점, 검정 배경 ---
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name=filename)
# opt = vis.get_render_option()
# opt.background_color = np.array([0, 0, 0])  # 검정 배경
# opt.point_size = 1.0                        # 포인트 크기 1
# vis.add_geometry(pcd)
# vis.run()
# vis.destroy_window()





# carla -> open3d 포인트 클라우드 바운딩박스 변환


# import open3d as o3d
# import numpy as np
# import os
# import cv2

# camera_path ='C:/Users/airlab/Carla-0.10.0/PythonAPI/examples/_out/00222889.png' #카메라 이미지
# lidar_path = 'C:/Users/airlab/Carla-0.10.0/PythonAPI/examples/_out/pc_222889.ply' #포인트클라우드
# label_path = 'C:/Users/airlab/Carla-0.10.0/PythonAPI/examples/_out/label/label_222889.txt' #라벨데이터 파일

# img = cv2.imread(camera_path)
# cv2.imshow("Camera View", img)
# cv2.waitKey(10)

# # 포인트클라우드 읽기
# pcd = o3d.io.read_point_cloud(lidar_path)

# #포인트 클라우드 y축반전
# temp =np.asarray(pcd.points)
# temp[:,1] = -temp[:,1]
# pcd.points = o3d.utility.Vector3dVector(temp[:, :3])

# num_points = np.asarray(pcd.points).shape[0]
# pcd.colors = o3d.utility.Vector3dVector(np.ones((num_points, 3)))  # 흰색

# pcd.translate([-95.490, -109.404 ,2.333])#포인트 클라우드 대칭이동(world좌표계로 봄)

# R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi* 121.295 /180))#회전변환
# center = [-95.490, -109.404 ,2.333]# 라이다 좌표를 중심으로(world좌표)
# #center = pcd2.get_center()
# pcd.rotate(R, center)#회전


# # 바운딩박스 읽기 함수
# def load_bboxes_from_label(label_path):
#     bboxes = []
#     with open(label_path, 'r') as f:
#         lines = f.readlines()

#     for line in lines:
#         if line.startswith('['):  # Header line
#             continue
#         if 'Location:' in line:
#             loc = list(map(float, line.strip().split(':')[1].split()))
#         if 'Rotation:' in line:
#             rot = list(map(float, line.strip().split(':')[1].split()))
#         if 'BBox Center:' in line:
#             center = list(map(float, line.strip().split(':')[1].split()))
#         if 'BBox Extent:' in line:
#             extent = list(map(float, line.strip().split(':')[1].split()))
#             # ✅ Open3D BoundingBox 생성
#             obb = o3d.geometry.OrientedBoundingBox()
#             center[1] = -center[1] #y값 반전 world->open3d
#             obb.center = center
#             obb.extent = np.array(extent)*2 # extend의 값은 1/2로 출력되어 2배
            
#             # 회전 행렬 생성 (yaw만 적용, Open3D는 z축 기준)
#             yaw = np.deg2rad(rot[1])  # rot[1] == yaw
#             R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, -yaw])# 회전 정보 filp 좌수->우수
#             obb.R = R
#             obb.color = [1, 0, 0]  # 빨간색
#             bboxes.append(obb)
#     return bboxes

# bboxes = load_bboxes_from_label(label_path)

# # ✅ 시각화
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="LiDAR + BBox 시각화")
# opt = vis.get_render_option()
# opt.background_color = np.array([0, 0, 0])
# opt.point_size = 1.0
# vis.add_geometry(pcd)
# for box in bboxes:
#     #box.translate([-62.013, -128.343, 2.491])
#     #ox.center[1]-=box.center[1]
#     R = box.get_rotation_matrix_from_xyz((0, 0, np.pi*121.295 /180))# 바운딩 박스 회전
#     center = box.get_center()
#     #box.rotate(R, center)
#     vis.add_geometry(box)
# lidar_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[-95.490, -109.404 ,2.333]) # 라이다 좌표축 그려주는 코드
# R = lidar_axis_pcd.get_rotation_matrix_from_xyz((0, 0, np.pi*121.295 /180)) # 회전정보 반영(world)
# #center = [-62.013, -128.343, 2.491]
# center = lidar_axis_pcd.get_center()
# lidar_axis_pcd.rotate(R, center)
# vis.add_geometry(lidar_axis_pcd)

# vehicle_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[-103.813, -99.460 ,-0.061]) # 자동차의 축을 그려주는 코드
# R = vehicle_axis_pcd.get_rotation_matrix_from_xyz((0, 0, np.pi* 110.624 /180))# 회전 정보를 반영
# #center = [-62.013, -128.343, 2.491]
# center = vehicle_axis_pcd.get_center()
# vehicle_axis_pcd.rotate(R, center)
# vis.add_geometry(vehicle_axis_pcd)

# world_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])
# vis.add_geometry(world_axis_pcd)    

# vis.run()
# vis.destroy_window()



#carla -> open3d  world 중심 시각화 코드

import open3d as o3d
import numpy as np
import os
import cv2

def visualize_all(base_dir):
    label_dir = os.path.join(base_dir, 'label')
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR + BBox + Axis",width=1080,height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 1.0

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0,0,0])
    vis.add_geometry(world_axis)

    def parse_lidar_ego_pose(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        lidar_origin, lidar_yaw, ego_origin, ego_yaw = None, None, None, None
        for idx, line in enumerate(lines):
            if '[LiDAR]' in line:
                loc = list(map(float, lines[idx+1].split(':')[1].split()))
                rot = list(map(float, lines[idx+2].split(':')[1].split()))
                lidar_origin, lidar_yaw = loc, rot[1]
            if '[Ego Vehicle]' in line:
                loc = list(map(float, lines[idx+1].split(':')[1].split()))
                rot = list(map(float, lines[idx+2].split(':')[1].split()))
                ego_origin, ego_yaw = loc, rot[1]
        return lidar_origin, lidar_yaw, ego_origin, ego_yaw

    def load_all_bboxes(label_file):
        bboxes = []
        with open(label_file, 'r') as f:
            lines = f.readlines()

        loc, rotation, center, extent = None, None, None, None

        for idx, line in enumerate(lines):
            if '[Ego Vehicle]' in line or '[Vehicle ID' in line:
                loc, rotation, center, extent = None, None, None, None

            if 'Location:' in line and ('[Ego Vehicle]' in lines[idx - 1] or '[Vehicle ID' in lines[idx - 1]):
                loc = list(map(float, line.strip().split(':')[1].split()))
            if 'Rotation:' in line and ('[Ego Vehicle]' in lines[idx - 2] or '[Vehicle ID' in lines[idx - 2]):
                rotation = list(map(float, line.strip().split(':')[1].split()))
            if 'BBox Center:' in line:
                center = list(map(float, line.strip().split(':')[1].split()))
            if 'BBox Extent:' in line:
                extent = list(map(float, line.strip().split(':')[1].split()))
                if loc and rotation and center and extent:
                    c = np.array(center)
                    c[1] *= -1
                    obb = o3d.geometry.OrientedBoundingBox()
                    obb.center = c
                    obb.extent = np.array(extent) * 2
                    yaw = np.deg2rad(rotation[1])
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, -yaw])
                    obb.R = R
                    obb.color = [1, 0, 0] if '[Vehicle ID' in lines[idx - 4] else [0, 1, 0]  # Vehicle 빨강, Ego 초록
                    bboxes.append(obb)
        return bboxes

    frame_ids = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(base_dir) if f.startswith('pc_')])

    for frame_id in frame_ids:
        lidar_file = os.path.join(base_dir, f'pc_{frame_id}.ply')
        label_file = os.path.join(label_dir, f'label_{frame_id}.txt')
        camera_file = os.path.join(base_dir, f'{str(frame_id).zfill(8)}.png')

        if os.path.exists(camera_file):
            img = cv2.imread(camera_file)
            cv2.imshow('Camera View', img)
            cv2.waitKey(10)

        lidar_origin, lidar_yaw, ego_origin, ego_yaw = parse_lidar_ego_pose(label_file)
        lidar_origin[1] *= -1
        ego_origin[1] *= -1
        pcd = o3d.io.read_point_cloud(lidar_file)
        pts = np.asarray(pcd.points)
        pts[:,1] *= -1
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts))

        pcd.translate(lidar_origin)
        lidar_R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi * -lidar_yaw / 180))
        pcd.rotate(lidar_R, center=lidar_origin)

        bboxes = load_all_bboxes(label_file)

        lidar_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=lidar_origin)
        lidar_axis.rotate(lidar_R, center=lidar_origin)

        ego_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=ego_origin)
        ego_R = ego_axis.get_rotation_matrix_from_xyz((0,0,np.pi * -ego_yaw/180))
        ego_axis.rotate(ego_R, center=ego_origin)

        vis.clear_geometries()
        vis.add_geometry(world_axis)
        vis.add_geometry(pcd)
        vis.add_geometry(lidar_axis)
        #vis.add_geometry(ego_axis)
        for box in bboxes:
            vis.add_geometry(box)
        vis.poll_events()
        vis.update_renderer()
        # set zoom, front, up, and lookat
        vis.get_view_control().set_zoom(0.3)
        vis.get_view_control().set_front([0, 0, 1])
        vis.get_view_control().set_up([1, 0, 0])
        vis.get_view_control().set_lookat(lidar_origin)
        vis.run()

    vis.destroy_window()
    cv2.destroyAllWindows()


# ✅ 실행
base_dir = 'C:/Users/airlab/Carla-0.10.0/PythonAPI/examples/_out'
visualize_all(base_dir)


