import csv
import numpy as np
import matplotlib.pyplot as plt

#file paths

FOCAL_FILE = "focal.txt"
JOINT_NAMES_FILE = "Pose/joint-names.txt"
POSES_FILE = "Post/poses.txt"

OUTPUT_CSV = "pose_2d_coordinates.csv"
IMG_WIDTH = 800
IMG_HEIGHT = 800

#load focal length

with open(FOCAL_FILE, "r") as f:
    focal_length = float(f.read().strip())

print("Focal length:", focal_length)

#load the joint names and ids

joint_ids = []
joint_names = []

with open(JOINT_NAMES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        j_id = int(parts[0])
        name = parts[1].strip("'")
        joint_ids.append(j_id)
        joint_names.append(name)

num_joints = len(joint_ids)
print("Num joints:", num_joints)
print("Joint names:", joint_names)

#load the poses

poses = []   #list of tuples of (camera_pos, joints_3d)

with open(POSES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vals = list(map(float, line.split()))
        expected = 3 + 3 * num_joints
        
        cam_pos = np.array(vals[0:3], dtype=np.float64)
        joints_flat = vals[3:]
        joints_3d = np.array(joints_flat, dtype=np.float64).reshape(num_joints, 3)
        poses.append((cam_pos, joints_3d))

num_frames = len(poses)
print("Num frames:", num_frames)

# 0-hip
# 1-rhip
# 2-rknee
# 3-rankle
# 4-lhip
# 5-lknee
# 6-lankle
# 7-neck
# 8-lupperarm
# 9-lelbow
# 10-lwrist
# 11-rupperarm
# 12-relbow
# 13-rwrist


edges = [
    (0, 1), (1, 2), (2, 3),         #right leg
    (0, 4), (4, 5), (5, 6),         #left leg
    (0, 7),                         #hip to neck
    (7, 8), (8, 9), (9, 10),        #left arm
    (7, 11), (11, 12), (12, 13)     #right arm
]

#create a camera rotation matrix by pointing the camera toward the subject
def build_look_at_rotation(cam_pos, joints_3d):
    # target: average of all joints
    target = joints_3d.mean(axis=0)

    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(z_axis, up_world)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(x_axis, z_axis)

    R = np.vstack((x_axis, y_axis, z_axis))
    return R

#project 3D joint coordinates into 2D image coordinates
def project_points(world_points, cam_pos, R, focal, img_w, img_h):
    shifted = world_points - cam_pos      #(N,3)
    cam_coords = (R @ shifted.T).T        #(N,3)

    X = cam_coords[:, 0]
    Y = cam_coords[:, 1]
    Z = cam_coords[:, 2]

    x_img = focal * (X / Z)
    y_img = focal * (Y / Z)

    pts = np.stack((x_img, y_img), axis=1)

    center = pts.mean(axis=0)
    pts_centered = pts - center

    max_abs = np.max(np.abs(pts_centered))
    if max_abs == 0:
        scale = 1.0
    else:
        scale = 0.4 * min(img_w, img_h) / max_abs

    pts_scaled = pts_centered * scale

    u = pts_scaled[:, 0] + img_w / 2.0
    v = -pts_scaled[:, 1] + img_h / 2.0   # flip y

    return np.stack((u, v), axis=1)





#loop over all frames: project joints, save CSV rows, and draw skeleton images
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "joint_id", "joint_name", "u", "v"])

    for frame_idx, (cam_pos, joints_3d) in enumerate(poses):
        R = build_look_at_rotation(cam_pos, joints_3d)
        joints_2d = project_points(
            joints_3d, cam_pos, R, focal_length, IMG_WIDTH, IMG_HEIGHT
        )

        # write coords
        for j_id, name, (u, v) in zip(joint_ids, joint_names, joints_2d):
            writer.writerow([frame_idx, j_id, name, u, v])

        # draw skeleton
        plt.figure(figsize=(4, 4))
        plt.scatter(joints_2d[:, 0], joints_2d[:, 1])

        for a, b in edges:
            x_vals = [joints_2d[a, 0], joints_2d[b, 0]]
            y_vals = [joints_2d[a, 1], joints_2d[b, 1]]
            plt.plot(x_vals, y_vals)

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()

        img_name = f"frame_{frame_idx:02d}.png"
        plt.savefig(img_name, dpi=150)
        plt.close()

        print("Saved", img_name)

print("Done. CSV:", OUTPUT_CSV)
