from utils import *
import polyscope as ps


def main():
    points, point_associations, normal_associations, trans = simulatePlanarScene()

    ps.init()
    ps.register_point_cloud("points", points)
    pc_associations = ps.register_point_cloud("point associations", point_associations)
    pc_associations.add_vector_quantity("normals", normal_associations, length=0.05, enabled=True)
    # Show the data association as lines
    all_points = np.concatenate([points, point_associations], axis=0)
    all_edges = np.concatenate([np.arange(len(points)), np.arange(len(points), len(points) + len(point_associations))]).reshape(2, -1).T
    ps.register_curve_network("associations", all_points, all_edges, color=(1.0, 0.0, 0.0), radius=0.001)
    ps.show()





if __name__ == "__main__":
    main()