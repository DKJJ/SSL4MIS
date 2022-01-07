import numpy as np
from sklearn import metrics
from PIL import Image

def get_metrics(pred, logits, gt):
    if isinstance(logits, list):
        logits = logits[-1]
    result = {'confusion_matrix': metrics.confusion_matrix(gt.flatten(), pred.flatten(), labels=[1, 0]),
              'auc': roc(gt, logits)}
    return result

def get_metrics_without_roc(pred, gt):
    result = {'confusion_matrix': metrics.confusion_matrix(gt.flatten(), pred.flatten(), labels=[1, 0])}
    return result

def show_metrics(metrics):
    con_mat = np.zeros((2,2))
    auc = 0.0
    for m in metrics:
        con_mat += m['confusion_matrix']
        auc += m['auc']
    auc /= len(metrics)
    result = {'confusion_matrix': con_mat.tolist(),
              'accuracy': accuracy(con_mat),
              'kappa': kappa(con_mat),
              'precision': precision(con_mat),
              'sensitivity': sensitivity(con_mat),
              'specificity': specificity(con_mat),
              'auc': auc,
              }
    return result

def show_metrics_without_roc(metrics):
    con_mat = np.zeros((2,2))
    for m in metrics:
        con_mat += m['confusion_matrix']
    result = {'confusion_matrix': con_mat,
              'accuracy': accuracy(con_mat),
              'kappa': kappa(con_mat),
              'precision': precision(con_mat),
              'sensitivity': sensitivity(con_mat),
              'specificity': specificity(con_mat),
              }
    return result

def show_metrics_from_save_image(data):
    pred = data[:,:,0] // 255
    gt = data[:,:,1] // 255
    metrics = [get_metrics_without_roc(pred, gt)]
    return show_metrics_without_roc(metrics)

def kappa(matrix):
    matrix = np.array(matrix)
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)


def sensitivity(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[1][0])


def specificity(matrix):
    return matrix[1][1]/(matrix[1][1]+matrix[0][1])


def precision(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[0][1])

def roc(gt, logits):
    gtlist = gt.flatten()
    predlist = logits.detach().cpu().numpy()[0, 1, ...].flatten()

    fpr, tpr, thresholds = metrics.roc_curve(gtlist, predlist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
    return roc_auc


def accuracy(matrix):
    return (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    return 100.0 - (
            100.0 *
            np.sum(np.argmin(predictions, 3) == np.argmin(labels, 3)) /
            (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))

def save_predict(filename, data, gt, pred):
    pred = pred * 255
    gt = gt[0, 1, :, :]
    gt = np.where(gt > 0.5, 255, 0)
    differ = np.stack([np.zeros_like(pred), gt, pred], -1)
    pred = np.stack([pred, pred, pred], -1)
    gt = np.stack([gt, gt, gt], -1)
    data = np.transpose(data, (0, 2, 3, 1))[0,...]
    if data.shape[2] == 60:
        data = data[:, :, 10:40:10]
    elif data.shape[2] == 1:
        data = np.concatenate([data, data, data], -1)
    elif data.shape[2] == 15:
        data = data[:, :, 0:15:5]
    data -= np.min(data, axis=(0,1))
    data /= (np.max(data, axis=(0,1))/255)
    data = data.astype(np.uint8)
    img = Image.fromarray(np.concatenate([data, pred, gt, differ], axis=1).astype(np.uint8))
    img.save(filename)

def save_logits(filename, pred):
    pred = pred * 255
    pred = np.stack([pred, pred, pred], -1)
    img = Image.fromarray(pred.astype(np.uint8))
    img.save(filename)
