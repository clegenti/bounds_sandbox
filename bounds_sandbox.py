from utils import *
import polyscope as ps
from scipy.spatial.transform import Rotation as R
import numpy as np

def main():
    boundsBoilerPlate()


def getRotationBounds(points, point_associations, normal_associations, rotation_estimate, point_stdev=0.03, normal_angle_stdev=3.0, num_samples=10000):
    normals_a, bounds_a = getPointCloudNormalsAndBounds(points, point_stdev)
    normals_b = normal_associations
    bounds_b = np.array([normalAngleStdevToBounds(normal_angle_stdev)]*len(normals_b))
    # Uncomment next line to test the case where the bounds are the same for both clouds (I guess lower than the bounds from the neighborhood approach)
    #bounds_a = np.copy(bounds_b)

    counter = 0
    min_bound = float('inf')
    while counter < num_samples:
        # Select 3 random indices
        indices = np.random.choice(len(points), size=3, replace=False)

        normals_a_sample = [normals_a[i] for i in indices]
        bounds_a_sample = [bounds_a[i] for i in indices]
        normals_b_sample = [normals_b[i] for i in indices]
        bounds_b_sample = [bounds_b[i] for i in indices]

        if(checkCollinearity(normals_a_sample) or checkCollinearity(normals_b_sample)):
            continue

        z = np.array([normalToNornalResiduals(normals_a_sample[i], normals_b_sample[i], rotation_estimate) + bounds_a_sample[i] + bounds_b_sample[i] for i in range(3)])
        A = np.zeros((3, 3))
        for i in range(3):
            A[:, i] = normals_a_sample[i]

        # Get the singular values of A
        U, S, Vt = np.linalg.svd(A)
        s2 = S[1]**2
        s3 = S[2]**2

        # Compute the bound on the rotation error using the formula derived in the paper
        bound = np.sqrt(2.0*np.linalg.norm(z) / (s2**2 + s3**2))

        min_bound = min(min_bound, bound)

        counter += 1
    return min_bound
    
    


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
    print(f"Rotation error (degrees): {np.degrees(rot_error):.4f}")
    print(f"Translation error: {trans_error:.4f}")

    ## Show the target and source points in polyscope
    #ps.init()
    #ps.register_point_cloud("points source", points)
    #pc_associations = ps.register_point_cloud("point targets", points_target)
    #pc_associations.add_vector_quantity("normals", normals_target, length=0.05, enabled=True)
    #ps.register_point_cloud("points source aligned", points_aligned)

    ## Show the data association as lines
    #all_points = np.concatenate([points, points_target], axis=0)
    #all_edges = np.concatenate([np.arange(len(points)), np.arange(len(points), len(points) + len(points_target))]).reshape(2, -1).T
    #ps.register_curve_network("associations", all_points, all_edges, color=(1.0, 0.0, 0.0), radius=0.001)
    #ps.show()


    # Get the bounds on the rotation estimate
    rotation_bounds = getRotationBounds(points, points_target, normals_target, T_target_source[:3, :3], point_stdev, normal_angle_stdev)

    print(f"Rotation bounds: {rotation_bounds:.4f}")
    print(f"Rotation bounds (degrees): {np.degrees(frobeniusToAngle(rotation_bounds)):.4f}")




if __name__ == "__main__":
    main()