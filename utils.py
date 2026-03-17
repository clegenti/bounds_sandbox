import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.spatial import KDTree


kNumOfSigmasForBounds = 3.0

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

def pointToPointRegistration(points, point_associations, max_iterations=100, tolerance=1e-6):
    # Initialize transformation
    pos_rot_vec = np.zeros(6)  # 3 for rotation (axis-angle), 3 for translation

    def costFunction(pos_rot_vec):
        rot = R.from_rotvec(pos_rot_vec[:3]).as_matrix()
        pos = pos_rot_vec[3:]
        transformed_points = (rot @ points.T).T + pos
        residuals = transformed_points - point_associations
        return residuals.flatten()
    
    result = least_squares(costFunction, pos_rot_vec, max_nfev=max_iterations, ftol=tolerance)
    final_rot = R.from_rotvec(result.x[:3]).as_matrix()
    final_pos = result.x[3:]

    pose = np.eye(4)
    pose[:3, :3] = final_rot
    pose[:3, 3] = final_pos
    return pose

def normalToNornalResiduals(n1, n2, rotation):
    res_1 = rotation @ n1 - n2
    res_2 = rotation @ n1 + n2
    if np.linalg.norm(res_1) < np.linalg.norm(res_2):
        return res_1
    else:
        return res_2


def simulatePlanarScene(num_planes=5, num_point_per_plane=100, point_noise_std=0.03, normal_angle_noise_std=3.0):

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





def skewSymmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def getAzElAndJacobian(normal):
    az = np.arctan2(normal[1], normal[0])
    el = np.arctan2(normal[2], np.sqrt(normal[0]**2 + normal[1]**2))
    
    d_az_d_normal = np.array([-normal[1]/(normal[0]**2 + normal[1]**2), normal[0]/(normal[0]**2 + normal[1]**2), 0])
    d_el_d_normal = np.array([-normal[0]*normal[2]/(normal[0]**2 + normal[1]**2 + normal[2]**2)/np.sqrt(normal[0]**2 + normal[1]**2), -normal[1]*normal[2]/(normal[0]**2 + normal[1]**2 + normal[2]**2)/np.sqrt(normal[0]**2 + normal[1]**2), np.sqrt(normal[0]**2 + normal[1]**2)/(normal[0]**2 + normal[1]**2 + normal[2]**2)])

    return np.array([az, el]), np.vstack((d_az_d_normal, d_el_d_normal))


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


    #temp = (np.eye(3) - np.outer(normal_unit, normal_unit)) / normal_norm

    #d_normal_unit_d_points = temp @ d_normal_d_points


    cov_points = np.zeros((9, 9))
    for i in range(3):
        cov_points[i*3:(i+1)*3, i*3:(i+1)*3] = np.eye(3) * point_stdevs[i]**2
        
    #cov_normal_unit = d_normal_unit_d_points @ cov_points @ d_normal_unit_d_points.T

    ## Get the bounds as the maximum eigenvalue of the covariance matrix
    #eigenvalues = np.linalg.eigvalsh(cov_normal_unit)
    #bound = kNumOfSigmasForBounds*np.sqrt(np.max(eigenvalues))


    az_el, d_az_el_d_normal = getAzElAndJacobian(normal)
    cov_az_el = d_az_el_d_normal @ d_normal_d_points @ cov_points @ d_normal_d_points.T @ d_az_el_d_normal.T
    eigenvalues_az_el = np.linalg.eigvalsh(cov_az_el)
    bound_az_el = kNumOfSigmasForBounds*np.sqrt(np.max(eigenvalues_az_el))
    bound = normalAngleStdevToBounds(bound_az_el)


    return normal_unit,bound



def normalAngleStdevToBounds(normal_angle_stdev):
    return np.linalg.norm(np.array([np.sin(np.radians(kNumOfSigmasForBounds*normal_angle_stdev)), 1 - np.cos(np.radians(kNumOfSigmasForBounds*normal_angle_stdev))]))

def getPointCloudNormalsAndBounds(points, point_stdev=0.03):
    normals = []
    bounds = []
    # Create a KDTree for efficient neighbor search
    tree = KDTree(points)
    for point in points:
        # Find the 3 nearest neighbors
        dists, idxs = tree.query(point, k=3)
        neighbors = [points[idx] for idx in idxs]
        normal, bound = pointsToNormalAndBounds(neighbors, point_stdevs=[point_stdev]*3)
        normals.append(normal)
        bounds.append(bound)
    return np.array(normals), np.array(bounds)

def checkCollinearity(normals, min_angle_deg= 5.0):
    min_angle_rad = np.radians(min_angle_deg)
    for i in range(len(normals)):
        for j in range(i+1, len(normals)):
            normal_b = normals[j]
            # Flip normal_b to have the smallest angle with normal_a
            if np.dot(normals[i], normal_b) < 0:
                normal_b = -normal_b
            angle = np.arccos(np.clip(np.dot(normals[i], normal_b)/ (np.linalg.norm(normals[i]) * np.linalg.norm(normal_b)), -1.0, 1.0))
            if angle < min_angle_rad or np.abs(angle - np.pi) < min_angle_rad:
                return True
    return False

def frobeniusToAngle(frobenius_norm):
    theta = 2*np.arcsin(frobenius_norm/(2*np.sqrt(2)))
    return theta


def getBoundFromSigma(sigma):
    return kNumOfSigmasForBounds*sigma