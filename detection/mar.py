from scipy.spatial import distance


def calculate_mar(mouth_points):

    A = distance.euclidean(mouth_points[1], mouth_points[7])

    B = distance.euclidean(mouth_points[2], mouth_points[6])

    C = distance.euclidean(mouth_points[3], mouth_points[5])

    D = distance.euclidean(mouth_points[0], mouth_points[4])

    mar = (A + B + C) / (2.0 * D)

    return mar