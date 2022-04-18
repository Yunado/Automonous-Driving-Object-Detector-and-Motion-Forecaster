import numpy as np
from shapely.geometry import Polygon
from shapely import affinity


def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    iou_mat = np.zeros((M, N))
    for m in range(M):
        box1 = bboxes1[m]
        cx1, cy1, l1, w1, theta1 = box1[0], box1[1], box1[2], box1[3], box1[4]
        ox1, oy1 = l1 / 2, w1 / 2
        for n in range(N):
            box2 = bboxes2[n]
            cx2, cy2, l2, w2, theta2 = box2[0], box2[1], box2[2], box2[3], box2[4]
            ox2, oy2 = l2 / 2, w2 / 2

            poly1_x1, poly1_x2, poly1_y1, poly1_y2 = cx1 - ox1, cx1 + ox1, cy1 - oy1, cy1 + oy1
            poly2_x1, poly2_x2, poly2_y1, poly2_y2 = cx2 - ox2, cx2 + ox2, cy2 - oy2, cy2 + oy2
            poly1 = Polygon([(poly1_x1, poly1_y1), (poly1_x1, poly1_y2),
                             (poly1_x2, poly1_y2), (poly1_x2, poly1_y1)])
            poly2 = Polygon([(poly2_x1, poly2_y1), (poly2_x1, poly2_y2),
                             (poly2_x2, poly2_y2), (poly2_x2, poly2_y1)])
            poly1 = affinity.rotate(poly1, theta1, use_radians=True)
            poly2 = affinity.rotate(poly2, theta2, use_radians=True)

            poly_intersection = poly1.intersection(poly2).area
            poly_union = poly1.union(poly2).area
            iou_mat[m, n] = poly_intersection / poly_union
    return iou_mat


def c_dist(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        c_dist: matrix of shape [M, N], where c_dist[i, j] is the 2D dist value between bboxes[i] and bboxes[j].
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    dist_mat = np.zeros((M, N))
    for m in range(M):
        box1 = bboxes1[m]
        cx1, cy1, l1, w1, theta1 = box1[0], box1[1], box1[2], box1[3], box1[4]
        ox1, oy1 = l1 / 2, w1 / 2
        for n in range(N):
            box2 = bboxes2[n]
            cx2, cy2, l2, w2, theta2 = box2[0], box2[1], box2[2], box2[3], box2[4]
            ox2, oy2 = l2 / 2, w2 / 2

            a = np.array([cx1 - ox1, cy1 - oy1])
            b = np.array([cx2 + ox2, cy2 + oy2])
            dist = np.linalg.norm(a - b)

            sqr = np.sqrt(np.power(cx1 - cx2, 2) + np.power(cy1 - cy2, 2))

            dist_mat[m, n] = sqr / dist
    return dist_mat


def c_shp(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        c_shp: matrix of shape [M, N], where c_shp[i, j] is the 2D shp value between bboxes[i] and bboxes[j].
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    shp_mat = np.zeros((M, N))
    for m in range(M):
        box1 = bboxes1[m]
        cx1, cy1, l1, w1, theta1 = box1[0], box1[1], box1[2], box1[3], box1[4]
        ox1, oy1 = l1 / 2, w1 / 2
        for n in range(N):
            box2 = bboxes2[n]
            cx2, cy2, l2, w2, theta2 = box2[0], box2[1], box2[2], box2[3], box2[4]
            ox2, oy2 = l2 / 2, w2 / 2

            poly1_x1, poly1_x2, poly1_y1, poly1_y2 = cx1 - ox1, cx1 + ox1, cy1 - oy1, cy1 + oy1
            poly2_x1, poly2_x2, poly2_y1, poly2_y2 = cx2 - ox2, cx2 + ox2, cy2 - oy2, cy2 + oy2
            poly1 = Polygon([(poly1_x1, poly1_y1), (poly1_x1, poly1_y2),
                             (poly1_x2, poly1_y2), (poly1_x2, poly1_y1)])
            poly2 = Polygon([(poly2_x1, poly2_y1), (poly2_x1, poly2_y2),
                             (poly2_x2, poly2_y2), (poly2_x2, poly2_y1)])
            poly1 = affinity.rotate(poly1, theta1, use_radians=True)
            poly2 = affinity.rotate(poly2, theta2, use_radians=True)

            sqr = np.sqrt(np.power(l1 - l2, 2) + np.power(w1 - w2, 2))
            shp = poly1.union(poly2).area

            shp_mat[m, n] = sqr / shp
    return shp_mat
