import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


def pointToPlaneRegistration(points, point_associations, normal_associations, max_iterations=100, tolerance=1e-6):
    # Initialize transformation
    pos_rot_vec = np.zeros(6)  # 3 for rotation (axis-angle), 3 for translation

    def costFunction(pos_rot_vec):
        rot = R.from_rotvec(pos_rot_vec[:3]).as_matrix()
        pos = pos_rot_vec[3:]
        transformed_points = (rot @ points.T).T + pos
        residuals = np.sum((transformed_points - point_associations) * normal_associations, axis=1)
        return residuals
    
    result = least_squares(costFunction, pos_rot_vec, max_nfev=max_iterations, ftol=tolerance)
    final_rot = R.from_rotvec(result.x[:3]).as_matrix()
    final_pos = result.x[3:]

    pose = np.eye(4)
    pose[:3, :3] = final_rot
    pose[:3, 3] = final_pos
    return pose



def simulatePlanarScene(num_planes=5, num_point_per_plane=100, point_noise_std=0.03, normal_angle_noise_std=3.0, seed=42):
    np.random.seed(42)

    plane_eqs = []
    for _ in range(num_planes):
        # Random normal vector
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal)

        # Random distance from the origin
        d = np.random.uniform(2.0, 15.0)

        plane_eqs.append((normal, d))


    points = []
    point_associations = []
    normal_associations = []
    for normal, d in plane_eqs:
        # Generate points on the plane
        for _ in range(num_point_per_plane):
            # Random point in the plane
            point = np.random.randn(3)
            point -= point.dot(normal) * normal  # Project onto the plane
            point += normal * d  # Shift to the correct distance

            point_asso = point + np.random.randn(3) * point_noise_std
            rand_rot = R.from_rotvec(np.random.randn(3) * np.radians(normal_angle_noise_std)).as_matrix()
            normal_asso = rand_rot @ normal
            point_associations.append(point_asso)
            normal_associations.append(normal_asso)


            # Add noise
            point += np.random.randn(3) * point_noise_std

            points.append(point)

    points = np.array(points)
    point_associations = np.array(point_associations)
    normal_associations = np.array(normal_associations)

    # Transform the points with a random rotation and translation
    rand_rot = R.from_rotvec(np.random.randn(3) * np.radians(30)).as_matrix()
    rand_pos = np.random.randn(3) * 5.0
    points = (rand_rot @ points.T).T + rand_pos

    trans = np.eye(4)
    trans[:3, :3] = rand_rot
    trans[:3, 3] = rand_pos

    return points, point_associations, normal_associations, np.linalg.inv(trans)

