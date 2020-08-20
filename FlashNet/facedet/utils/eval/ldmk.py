import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# from scipy.interpolate import spline
from scipy.interpolate import interp1d
import numpy as np
from sklearn.metrics import auc

def calc_error_rate_i(pred_landmarks, gt_landmarks, error_normalize_factor, num_landmark=5):
    error = []

    # print('pred_landmarks', pred_landmarks)
    # print('gt_landmarks', gt_landmarks)

    for i in range(num_landmark):
        # import pdb
        # pdb.set_trace()
        error.append(np.sqrt(np.power(pred_landmarks[i] - gt_landmarks[i],2).sum()))
    # print('error', error)
    return sum(error)/num_landmark/error_normalize_factor


def calc_auc(total_img, error_rate, max_threshold):
    error_rate = np.array(error_rate)
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_rate < threshold[i]) * 1.0 / total_img
    return auc(threshold, accuracys) / max_threshold, accuracys



def eval_CED(auc_record):
    error = np.linspace(0., 0.1, 21)
    error_new = np.linspace(error.min(), error.max(), 300)
    auc_value = np.array([auc_record[0], auc_record[99], auc_record[199], auc_record[299],
                          auc_record[399], auc_record[499], auc_record[599], auc_record[699],
                          auc_record[799], auc_record[899], auc_record[999], auc_record[1099],
                          auc_record[1199], auc_record[1299], auc_record[1399], auc_record[1499],
                          auc_record[1599], auc_record[1699], auc_record[1799], auc_record[1899],
                          auc_record[1999]])
    CFSS_auc_value = np.array([0., 0., 0., 0., 0.,
                               0., 0.02, 0.09, 0.18, 0.30,
                               0.45, 0.60, 0.70, 0.75, 0.79,
                               0.82, 0.85, 0.87, 0.88, 0.89, 0.90])
    SAPM_auc_value = np.array([0., 0., 0., 0., 0.,
                               0., 0., 0., 0.02, 0.08,
                               0.17, 0.28, 0.43, 0.58, 0.71,
                               0.78, 0.83, 0.86, 0.89, 0.91, 0.92])
    TCDCN_auc_value = np.array([0., 0., 0., 0., 0.,
                                0., 0., 0.02, 0.05, 0.10,
                                0.19, 0.29, 0.38, 0.47, 0.56,
                                0.64, 0.70, 0.75, 0.79, 0.82, 0.826])
    # auc_smooth = spline(error, auc_value, error_new)
    # CFSS_auc_smooth = spline(error, CFSS_auc_value, error_new)
    # SAPM_auc_smooth = spline(error, SAPM_auc_value, error_new)
    # TCDCN_auc_smooth = spline(error, TCDCN_auc_value, error_new)
    #
    #
    # plt.plot(error_new, auc_smooth, 'r-')
    # plt.plot(error_new, CFSS_auc_smooth, 'g-')
    # plt.plot(error_new, SAPM_auc_smooth, 'y-')
    # plt.plot(error_new, TCDCN_auc_smooth, 'm-')

    f=interp1d(error, auc_value, kind='cubic')
    f_CFSS=interp1d(error, CFSS_auc_value, kind='cubic')
    f_SAPM=interp1d(error, SAPM_auc_value, kind='cubic')
    f_TCDCN=interp1d(error, TCDCN_auc_value, kind='cubic')

    plt.plot(error_new, f(error_new), 'r-')
    plt.plot(error_new, f_CFSS(error_new), 'g-')
    plt.plot(error_new, f_SAPM(error_new), 'y-')
    plt.plot(error_new, f_TCDCN(error_new), 'm-')

    plt.legend(['Ours, Error: 5.35%, Failure: 4.27%',
                'CFSS, Error: 6.28%, Failure: 9.07%',
                'SAPM, Error: 6.64%, Failure: 5.72%',
                'TCDCN, Error: 7.66%, Failure: 16.17%'], loc=4)
    plt.plot(error, auc_value, 'rs')
    plt.plot(error, CFSS_auc_value, 'go')
    plt.plot(error, SAPM_auc_value, 'y^')
    plt.plot(error, TCDCN_auc_value, 'mx')
    plt.axis([0., 0.1, 0., 1.])
    # import pdb
    # pdb.set_trace()
    # plt.show()
    plt.gcf().savefig('ced.jpg')
