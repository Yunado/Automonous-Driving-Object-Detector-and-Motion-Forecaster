import numpy as np

from tracking.cost import iou_2d


def test_iou_2d() -> None:
    # test 1
    bboxes_a = np.array([[0.0, 0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 2.0, 1.0, np.pi / 2]])
    bboxes_b = np.array([[1.0, 0.0, 2.0, 1.0, 0.0], [1.0, 0.0, 2.0, 1.0, np.pi / 2]])
    iou_mat = iou_2d(bboxes_a, bboxes_b)
    exp_iou_mat = np.array([[1 / 3.0, 0.5 / 3.5], [0.5 / 3.5, 0.0]])
    np.testing.assert_allclose(iou_mat, exp_iou_mat, atol=1e-5)
    print("test 1 passed")

    # test 2
    iou_mat = iou_2d(bboxes_a, bboxes_a)
    exp_iou_mat = np.array([[1.0, 1 / 3.0], [1 / 3.0, 1.0]])
    np.testing.assert_allclose(iou_mat, exp_iou_mat, atol=1e-5)
    print("test 2 passed")

    # test 3
    bboxes_a = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )
    bboxes_b = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, np.pi / 4],
        ]
    )
    iou_mat = iou_2d(bboxes_a, bboxes_b)
    exp_iou_mat = (1 - 2 * (1 - 0.5 * np.sqrt(2)) ** 2) / (
        1 + 2 * (1 - 0.5 * np.sqrt(2)) ** 2
    )
    np.testing.assert_allclose(iou_mat, exp_iou_mat)
    print("test 3 passed")

    bboxes_a = np.array([[0.0, 0.0, 2.0, 1.0, 0.0]])
    bboxes_b = np.array([[1.0, 0.0, 2.0, 1.0, 0.0], [1.0, 0.0, 2.0, 1.0, np.pi / 2]])
    iou_mat = iou_2d(bboxes_a, bboxes_b)
    exp_iou_mat = np.array([[1 / 3.0, 0.5 / 3.5]])
    np.testing.assert_allclose(iou_mat, exp_iou_mat, atol=1e-5)

def test_more() -> None:
    # test 1
    bboxes_a = np.array([[2.0, 2.0, 4.0, 4.0, 0.0], [4.0, 4.0, 5, 1, np.pi/3]])
    bboxes_b = np.array([[3.0, 3.0, 2.0, 2.0, 0.0], [6.0, 6.0, 4.0, 4.0, 0.0]])
    iou_mat = iou_2d(bboxes_a, bboxes_b)
    cost_mat = np.ones((2, 2)) - iou_mat
    exp_iou_mat = np.array([[0.25, 0], [0.288, 0.1177]])
    print(iou_mat)
    print(cost_mat)
    np.testing.assert_allclose(iou_mat, exp_iou_mat, atol=1e-5)
