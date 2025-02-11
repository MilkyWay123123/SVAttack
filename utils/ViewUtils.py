import numpy as np
import torch


def look_at(eye, target, up):
    eye = torch.tensor(eye, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)

    forward = target - eye
    forward = forward / torch.norm(forward)

    right = torch.cross(forward, up)
    right = right / torch.norm(right)

    up = torch.cross(right, forward)

    view_matrix = torch.stack([
        torch.cat([right, torch.tensor([-torch.dot(right, eye)])]),
        torch.cat([up, torch.tensor([-torch.dot(up, eye)])]),
        torch.cat([-forward, torch.tensor([torch.dot(forward, eye)])]),
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)
    ], dim=0)

    return view_matrix


def perspective(fov, aspect, near, far):
    f = 1.0 / torch.tan(fov / 2.0)
    depth = near - far

    proj_matrix = torch.tensor([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / depth, (2 * far * near) / depth],
        [0, 0, -1, 0]
    ], dtype=torch.float32)

    return proj_matrix


def project(points, view_matrix, proj_matrix):
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    points = torch.cat((points, ones), dim=-1)
    view_points = torch.matmul(points, view_matrix.t())
    proj_points = torch.matmul(view_points, proj_matrix.t())
    proj_points = proj_points / proj_points[..., 3].unsqueeze(-1)
    return proj_points[..., :2]


'''
The position of the camera is calculated based on the set elevation and azimuth angles
'''
def compute_angle(angle_elevation=35.26, angle_azimuth=45):
    # Define new elevation and azimuth angles (in degrees)
    elev_deg = angle_elevation
    azim_deg = angle_azimuth

    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    # Calculate the distance r (keep it the same)
    r = np.sqrt(3.0 ** 2 + 3.0 ** 2 + 3.0 ** 2)

    # Calculate the new camera position
    x_new = r * np.cos(elev) * np.cos(azim)
    y_new = r * np.cos(elev) * np.sin(azim)
    z_new = r * np.sin(elev)

    eye_new = np.array([x_new, y_new, z_new])

    return eye_new

def ThreeDimensionsToTwoDimensions(data, elev=None, azim=None):
    device = data.device
    N, C, T, V, M = data.shape

    if elev != None:
        eye = compute_angle(elev, azim)
    else:
        eye = [3.0, 3.0, 3.0]
    target = [0.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]

    fov = torch.deg2rad(torch.tensor(90.0))
    aspect = 1.0
    near = 0.1
    far = 100.0

    view_matrix = look_at(eye, target, up)
    view_matrix = view_matrix.to(device)

    proj_matrix = perspective(fov, aspect, near, far)
    proj_matrix = proj_matrix.to(device)

    data = data.permute(0, 2, 3, 4, 1).contiguous()
    data_reshaped = data.view(-1, C)
    points_2d = project(data_reshaped, view_matrix, proj_matrix)
    output_data = points_2d.view(N, T, V, M, 2)
    output_data = output_data.permute(0, 4, 1, 2, 3)

    return output_data
