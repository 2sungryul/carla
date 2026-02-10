import carla
import matplotlib.pyplot as plt
import numpy as np
import os

def carla_to_ros2(x, y):
    """
    CARLA 좌표계를 ROS2 좌표계로 변환
    CARLA: x(forward), y(right), z(up)
    ROS2: x(forward), y(left), z(up)
    따라서 y축만 반전
    """
    return x, -y

def main():
    # CARLA 서버에 연결
    print("CARLA 서버에 연결 중...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # 현재 맵 가져오기
    world = client.get_world()
    carla_map = world.get_map()
    map_name = carla_map.name
    
    print(f"현재 로드된 맵: {map_name}")
    
    # 맵 이름 추출 (파일명 생성용)
    map_simple_name = map_name.split('/')[-1]
    
    print("웨이포인트 생성 중...")
    
    # 모든 웨이포인트 가져오기 - 간격을 더 짧게 설정
    waypoints = carla_map.generate_waypoints(1.0)  # 1m 간격으로 더 세밀하게
    
    # 토폴로지 기반으로 모든 차선 가져오기
    print("토폴로지 기반 웨이포인트 생성 중...")
    topology = carla_map.get_topology()
    topology_waypoints = []
    for wp_pair in topology:
        topology_waypoints.extend(wp_pair)
    
    # 두 가지 방법으로 얻은 웨이포인트 합치기
    all_waypoints = list(waypoints) + list(topology_waypoints)
    
    # Spawn point 가져오기
    print("Spawn point 가져오는 중...")
    spawn_points = carla_map.get_spawn_points()
    
    # Spawn point 좌표 변환
    spawn_positions = []
    for i, spawn in enumerate(spawn_points):
        x_ros, y_ros = carla_to_ros2(spawn.location.x, spawn.location.y)
        spawn_positions.append((x_ros, y_ros, i))
    
    # 차선별로 웨이포인트 그룹화
    lanes = {}
    waypoint_positions = []
    waypoint_ids = []
    
    print(f"총 {len(all_waypoints)}개의 웨이포인트 처리 중...")
    
    for i, wp in enumerate(all_waypoints):
        # CARLA 좌표를 ROS2 좌표로 변환
        x_ros, y_ros = carla_to_ros2(wp.transform.location.x, wp.transform.location.y)
        waypoint_positions.append((x_ros, y_ros))
        waypoint_ids.append(i)
        
        # 차선 ID 생성 (road_id, section_id, lane_id 조합)
        lane_id = (wp.road_id, wp.section_id, wp.lane_id)
        
        if lane_id not in lanes:
            lanes[lane_id] = []
        lanes[lane_id].append((x_ros, y_ros))
    
    print(f"총 {len(lanes)}개의 고유한 차선 발견")
    
    # 시각화
    plt.figure(figsize=(20, 20))
    
    # 각 차선 그리기 - 색상을 다양하게 하여 차선 구분
    print(f"차선 시각화 중... (총 {len(lanes)}개 차선)")
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(lanes))))
    
    for idx, (lane_id, points) in enumerate(lanes.items()):
        if len(points) > 1:
            # 차선의 연속성을 위해 정렬
            points = sorted(points, key=lambda p: (p[0], p[1]))
            xs, ys = zip(*points)
            color = colors[idx % len(colors)] if len(lanes) > 1 else 'b'
            plt.plot(xs, ys, '-', color=color, linewidth=1.0, alpha=0.7, 
                    label=f'Lane {lane_id[2]}' if idx < 10 else "")  # 범례는 처음 10개만
    
    """
    # 경유점(웨이포인트) 그리기
    for (x, y), wp_id in zip(waypoint_positions, waypoint_ids):
        plt.plot(x, y, 'ro', markersize=2, alpha=0.3)
        # 모든 경유점 번호를 표시하면 너무 복잡하므로 일부만 표시
        if wp_id % 50 == 0:  # 50개마다 하나씩만 번호 표시
            plt.text(x, y, str(wp_id), fontsize=6, alpha=0.5)
    """
    
    # Spawn point 그리기
    print(f"Spawn point 그리는 중... (총 {len(spawn_positions)}개)")
    for x, y, spawn_id in spawn_positions:
        # Spawn point를 큰 원으로 표시
        plt.plot(x, y, 'mo', markersize=8, alpha=0.7, markeredgewidth=1.5, 
                markeredgecolor='purple', markerfacecolor='yellow', zorder=3)
        # Spawn point 번호 표시
        plt.text(x, y+5, str(spawn_id), fontsize=8, fontweight='bold', 
                color='purple', ha='center', va='bottom', zorder=3)
    
    # 원점 표시
    plt.plot(0, 0, 'g*', markersize=15, label='Origin (CARLA & ROS2)', zorder=5)
    
    # 축 설정 (ROS2 좌표계)
    plt.xlabel('X (ROS2) [m]', fontsize=12)
    plt.ylabel('Y (ROS2) [m]', fontsize=12)
    plt.title(f'CARLA {map_simple_name} - Road Lanes (ROS2 Coordinate System)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    # 좌표축 눈금 설정
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    
    # 좌표계 표시를 위한 화살표 추가
    arrow_length = 25
    plt.arrow(0, 0, arrow_length, 0, head_width=10, head_length=10, 
              fc='red', ec='red', linewidth=2, label='X-axis (ROS2)', zorder=4)
    plt.arrow(0, 0, 0, arrow_length, head_width=10, head_length=10, 
              fc='green', ec='green', linewidth=2, label='Y-axis (ROS2)', zorder=4)
    
    # 저장 파일명 생성 (맵 이름 기반)
    output_filename = f'lanes_ros2.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n맵이 성공적으로 생성되었습니다!")
    print(f"저장 위치: {output_filename}")
    print(f"총 경유점 수: {len(waypoint_positions)}")
    print(f"총 차선 수: {len(lanes)}")
    print(f"총 Spawn Point 수: {len(spawn_positions)}")
    
    # 화면에 출력
    print("\n화면에 맵을 표시합니다...")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n주의사항:")
        print("1. CARLA 시뮬레이터가 실행 중이어야 합니다")
        print("2. CARLA Python API가 설치되어 있어야 합니다")
        print("3. CARLA 서버가 localhost:2000에서 실행 중이어야 합니다")