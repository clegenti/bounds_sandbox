from utils import *
import polyscope as ps
from scipy.spatial.transform import Rotation as R
import numpy as np

def main():
    point_test = []
    for i in range(3):
        point_test.append(np.random.randn(3))    
    pointsToNormalAndBounds(point_test)

    boundsBoilerPlate()



def normalAngleStdevToBounds(normal_angle_stdev):
    return np.linalg.norm(np.array([np.sin(np.radians(3.0*normal_angle_stdev)), 1 - np.cos(np.radians(3.0*normal_angle_stdev))]))


#def getRotationBounds(points, point_associations, normal_associations, point_stdev=0.03, normal_angle_stdev=3.0, num_samples=1000):

    

def pointsToNormalAndBounds(points, point_stdevs = [0.03]*3):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    normal_unit = normal / np.linalg.norm(normal)

    d_normal_d_point_0 = skewSymmetric(points[2] - points[1])
    d_normal_d_point_1 = -skewSymmetric(v2)
    d_normal_d_point_2 = skewSymmetric(v1)

    d_normal_d_points = np.zeros((3, 9))
    d_normal_d_points[:, :3] = d_normal_d_point_0
    d_normal_d_points[:, 3:6] = d_normal_d_point_1
    d_normal_d_points[:, 6:9] = d_normal_d_point_2


    temp = (np.eye(3) - np.outer(normal_unit, normal_unit)) / normal_norm

    d_normal_unit_d_points = temp @ d_normal_d_points


    cov_points = np.zeros((9, 9))
    for i in range(3):
        cov_points[i*3:(i+1)*3, i*3:(i+1)*3] = np.eye(3) * point_stdevs[i]**2
    cov_normal_unit = d_normal_unit_d_points @ cov_points @ d_normal_unit_d_points.T

    # Get the bounds as the maximum eigenvalue of the covariance matrix
    eigenvalues = np.linalg.eigvalsh(cov_normal_unit)
    bound = 3*np.sqrt(np.max(eigenvalues))

    return normal_unit,bound



def boundsBoilerPlate():
    point_stdev = 0.03
    normal_angle_stdev = 3.0
    points, points_target, normals_target, trans = simulatePlanarScene(num_planes=5, num_point_per_plane=100, point_noise_std=point_stdev, normal_angle_noise_std=normal_angle_stdev)

    # Perform point to plane registration
    T_target_source = pointToPlaneRegistration(points, points_target, normals_target)

    # Transform the source points to the target frame
    points_aligned = (T_target_source[:3, :3] @ points.T).T + T_target_source[:3, 3]

    # Get error metrics
    pose_diff = np.linalg.inv(T_target_source) @ trans
    rot_error = np.linalg.norm(R.from_matrix(pose_diff[:3, :3]).as_rotvec())
    trans_error = np.linalg.norm(pose_diff[:3, 3])
    print(f"Rotation error (radians): {rot_error:.4f}")
    print(f"Translation error: {trans_error:.4f}")

    # Show the target and source points in polyscope
    ps.init()
    ps.register_point_cloud("points source", points)
    pc_associations = ps.register_point_cloud("point targets", points_target)
    pc_associations.add_vector_quantity("normals", normals_target, length=0.05, enabled=True)
    ps.register_point_cloud("points source aligned", points_aligned)

    # Show the data association as lines
    all_points = np.concatenate([points, points_target], axis=0)
    all_edges = np.concatenate([np.arange(len(points)), np.arange(len(points), len(points) + len(points_target))]).reshape(2, -1).T
    ps.register_curve_network("associations", all_points, all_edges, color=(1.0, 0.0, 0.0), radius=0.001)
    ps.show()


    




if __name__ == "__main__":
    main()