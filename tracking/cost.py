import numpy as np
from shapely.geometry import Polygon


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
        ox1, oy1 = l1/2, w1/2
        for n in range(N):
            box2 = bboxes2[n]
            cx2, cy2, l2, w2, theta2 = box2[0], box2[1], box2[2], box2[3], box2[4]
            ox2, oy2 = l2/2, w2/2
            poly1_c1x = cx1 + ox1 * np.cos(theta1) - oy1 * np.sin(theta1)
            poly1_c1y = cy1 + ox1 * np.sin(theta1) + oy1 * np.cos(theta1)
            poly1_c2x = cx1 - ox1 * np.cos(theta1) - oy1 * np.sin(theta1)
            poly1_c2y = cy1 - ox1 * np.sin(theta1) + oy1 * np.cos(theta1)
            poly1_c3x = cx1 - ox1 * np.cos(theta1) + oy1 * np.sin(theta1)
            poly1_c3y = cy1 - ox1 * np.sin(theta1) - oy1 * np.cos(theta1)
            poly1_c4x = cx1 + ox1 * np.cos(theta1) + oy1 * np.sin(theta1)
            poly1_c4y = cy1 + ox1 * np.sin(theta1) - oy1 * np.cos(theta1)

            poly2_c1x = cx2 + ox2 * np.cos(theta2) - oy2 * np.sin(theta2)
            poly2_c1y = cy2 + ox2 * np.sin(theta2) + oy2 * np.cos(theta2)
            poly2_c2x = cx2 - ox2 * np.cos(theta2) - oy2 * np.sin(theta2)
            poly2_c2y = cy2 - ox2 * np.sin(theta2) + oy2 * np.cos(theta2)
            poly2_c3x = cx2 - ox2 * np.cos(theta2) + oy2 * np.sin(theta2)
            poly2_c3y = cy2 - ox2 * np.sin(theta2) - oy2 * np.cos(theta2)
            poly2_c4x = cx2 + ox2 * np.cos(theta2) + oy2 * np.sin(theta2)
            poly2_c4y = cy2 + ox2 * np.sin(theta2) - oy2 * np.cos(theta2)

            poly1 = Polygon([(poly1_c1x, poly1_c1y), (poly1_c2x, poly1_c2y),
                             (poly1_c3x, poly1_c3y), (poly1_c4x, poly1_c4y)])
            poly2 = Polygon([(poly2_c1x, poly2_c1y), (poly2_c2x, poly2_c2y),
                             (poly2_c3x, poly2_c3y), (poly2_c4x, poly2_c4y)])
            poly_intersection = poly1.intersection(poly2).area
            poly_union = poly1.union(poly2).area
            iou_mat[m, n] = poly_intersection/poly_union
    # print("this is iou mat")
    # print(iou_mat)
    # print("===============")
    return iou_mat
