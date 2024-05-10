import cv2
import numpy as np

def eye_in_hand_dorna_ta_embeded_camera(robot, kinematic, camera, charuco_board):
    # init window
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)

    # init data
    collect_data = {"joint": [], "R_target_2_camera": [], "t_target_2_camera": [], "id": []}

    # initialization
    R_target_2_camera = []
    t_target_2_camera = []
    charuco_id = [] 
    
    # wait to capture image
    #init_joint = np.array([317.417, 0, 120.662, -90, 0, 0])

    # Generate all possible lists of size 5
    #all_lists = np.array(list(itertools.product([-30, -10, 0, 10, 30], [-30, -10, 0, 10, 30], [-30, -10, 0, 10, 30], [0], [-10, 0, 10,])))
    #all_points = [init_point +  xyzab for xyzab in all_lists]
    counter = 0
    while True:
        #print("remaining points: ", len(all_points))
        #point = all_points.pop()
        #robot.lmove(rel=0, x=point[0],  y=point[1], z=point[2], a=point[3],  b=point[4])
        #robot.sleep(0.5)
        counter += 1

        # capture image
        _, _, _, _, _, color_img, depth_int, _, _= camera.get_all()

        # pose
        rvec, tvec, charuco_id, img_gray = charuco_board.pose(color_img, camera.camera_matrix(depth_int), camera.dist_coeffs(depth_int))   

        # show axis
        cv2.imshow("color",color_img)
        
        # wait 1ms
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'): # press q to exit
            break

        elif key & 0xFF == ord('s'): # press enter to record a position
            if not np.all(tvec == np.zeros((3, 1))):
                R_target_2_camera.append(rvec)
                t_target_2_camera.append(tvec)
                charuco_id.append(charuco_id)

        else:
            pass

    # run detection    
    #R_cam_2_j4, t_cam_2_j4 = cv2.calibrateHandEye(R_j4_2_base, t_j4_2_base, R_target_2_cam, t_target_2_cam)
    
    return 0


def main():
    from camera import Camera
    from board import Charuco, Aruco
    # camera
    camera = Camera()
    camera.connect()

    # board
    charuco_board = Charuco(8, 8, 12, 8, dictionary="DICT_6X6_250", refine="CORNER_REFINE_NONE", subpix=False)
    
    # pose
    eye_in_hand_dorna_ta_embeded_camera(None, None, camera, charuco_board)


    camera.close()


if __name__ == '__main__':
    main()