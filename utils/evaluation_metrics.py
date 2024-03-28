import math
import numpy as np

eps = 0.00001
x_axis_threshold = 0.9
y_axis_threshold = 0.75


def compute_nme(gt_data_batch, predicted_data, roi=None, gt_scale=1, norm_const_type='approx_roi', individual_nme=False):
    """
    Compute Normalized Mean Error
    Parameters
    ----------

    individual_nme : Bool
    norm_const_type : str
    gt_data_batch :  np.ndarray
    predicted_data : np.ndarray
    roi :   np.ndarray
    gt_scale: int

    Returns
    -------
    float   normalized mean error

    """

    gt_data_batch *= gt_scale

    if gt_data_batch.ndim == 2:
        gt_data_batch = gt_data_batch[np.newaxis, :]
        predicted_data = predicted_data[np.newaxis, :]
        if roi is not None:
            roi = roi[np.newaxis, :]

    if gt_data_batch.ndim < 2 or gt_data_batch.ndim > 3:
        raise ValueError('Ground-truth data and predictions should be wither 2D or 3D')

    number_landmarks = gt_data_batch.shape[1]

    if individual_nme:
        prediction_error = np.sqrt(np.sum((gt_data_batch - predicted_data) ** 2, axis=-1))
        nme = np.empty((gt_data_batch.shape[0], number_landmarks))
    else:
        euclidean_dist = np.sqrt(np.sum((gt_data_batch - predicted_data) ** 2, axis=-1))
        prediction_error = np.mean(euclidean_dist, axis=-1)
        nme = np.empty(gt_data_batch.shape[0])

    for index, gt_data in enumerate(gt_data_batch):
        if 'iod' in norm_const_type:
            if number_landmarks == 68:
                interocular_dist = np.linalg.norm(gt_data[36] - gt_data[45])
            elif number_landmarks == 5:
                interocular_dist = np.linalg.norm(gt_data[0] - gt_data[1])
            else:
                raise NotImplementedError('Number of landmarks must be eiter 68 or 5')
            normalizing_const = interocular_dist
        elif 'approx_roi' in norm_const_type:
            record = np.zeros((4,), dtype=np.float32)
            record[0:2] = np.min(gt_data, axis=0)
            record[2:4] = np.max(gt_data, axis=0)
            normalizing_const = np.sqrt((record[2] - record[0]) * (record[3] - record[1]))
        elif 'roi' in norm_const_type:
            roi_height = roi[index][3] - roi[index][1]
            roi_width = roi[index][2] - roi[index][0]
            normalizing_const = math.sqrt(roi_height * roi_width)
        else:
            return

        nme[index] = prediction_error[index] / (normalizing_const + eps)
    return np.squeeze(nme)


def compute_ced(errors, threshold_error_value=0.3, bin_size=5000):
    errors = errors.flatten()
    hist, bin_edges = np.histogram(errors, bins=bin_size, range=(0, threshold_error_value))

    cdf = np.cumsum(hist) / len(errors)

    auc = np.sum(np.diff(bin_edges) * cdf)

    return cdf, auc, bin_edges


def compute_rmse(gt_data, predicted_data):
    nominator = ((gt_data - predicted_data) ** 2).sum(dim=2)
    denominator = np.array(nominator.size()).prod()
    return np.sqrt(nominator.sum().item() / denominator)


def compute_l1_error(user_vectors, curr_user_vector):
    return np.sum(abs(user_vectors - curr_user_vector[np.newaxis]), axis=1)


def compute_mse(vector_1, vector_2):
    return np.sum((vector_1 - vector_2) ** 2)



