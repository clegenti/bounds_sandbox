from utils import *
import polyscope as ps
from scipy.spatial.transform import Rotation as R


def main():
    points, points_target, normals_target, trans = simulatePlanarScene()

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