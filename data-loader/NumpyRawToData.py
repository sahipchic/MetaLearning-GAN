import numpy as np
from os import walk
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pathlib import Path
import pandas as pd
import category_encoders as ce
from numpy import unravel_index


def load(name: str) -> (np.ndarray, np.ndarray):
    name_path = f'{raw_data_str}{name}/'
    data_path = f'{name_path}{name}_data.npy'
    class_path = f'{name_path}{name}_class.npy'
    data_np = np.load(data_path)
    class_np = np.load(class_path)
    return data_np, class_np


def prepareLabels(labels_in: np.ndarray) -> np.ndarray:
    onehot = OneHotEncoder(sparse=False, categories='auto')
    result = onehot.fit_transform(labels_in)
    return result


# def makeFeatures(labels_size: int, features_in: np.ndarray) -> np.ndarray:
#     (_, features_in_size) = features_in.shape
#     if features_in_size * (features_in_size - 1) / 2 < target_size - labels_size:
#         degree = 10
#         interaction_only = False
#     else:
#         degree = 2
#         interaction_only = True
#     poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
#     new_features = poly.fit_transform(features_in)
#     result = new_features[:, :target_size - labels_size]
#     assert result.shape[1] == target_size - labels_size
#     return result

def scaleData(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    data = np.append(zero_in, one_in, axis=0)
    data = data.astype(np.float64)
    min_max_scaler = MinMaxScaler()
    data_out = min_max_scaler.fit_transform(data)
    return data_out[:target_instances], data_out[target_instances:]


def shrinkFeatures(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=239)
    data = np.append(zero_in, one_in, axis=0)
    labels = np.append(np.zeros(target_instances), np.ones(target_instances))
    clf.fit(data, labels)
    model = SelectFromModel(estimator=clf, threshold=-np.inf, prefit=True, max_features=target_features)
    new_data = model.transform(data)
    if new_data.shape[1] == target_features:
        return new_data[:target_instances], new_data[target_instances:]
    else:
        given_features = data.shape[1]
        new_data = data[:, np.random.choice(given_features.shape[1], target_features, replace=False)]
        assert new_data.shape[1] == target_features
        return new_data[:target_instances], new_data[target_instances:]


def swapCol(data_in: np.array, frm: int, to: int) -> np.ndarray:
    data_in[:, [frm, to]] = data_in[:, [to, frm]]
    return data_in


def swapRow(data_in: np.array, frm: int, to: int) -> np.ndarray:
    data_in[[frm, to], :] = data_in[[to, frm], :]
    return data_in


def swapRows(data_in: np.array, frm: np.ndarray, to: np.ndarray, col_num: int) -> np.ndarray:
    to_sorted = np.argsort(data_in[to, col_num])[::-1]
    for i, j in enumerate(to_sorted):
        frm_i = frm[i]
        to_i = to[j]
        data_in[[frm_i, to_i], :] = data_in[[to_i, frm_i], :]
    return data_in


def findMax(zero_in: np.ndarray, one_in: np.ndarray, step: int) -> (int, np.ndarray, np.ndarray):
    max_col_mean = []
    (rows, cols) = zero_in.shape
    for i in range(cols):
        zero_col = zero_in[:, i]
        zero_max_ind = np.argpartition(zero_col, -step)[-step:]
        if rows != step:
            zero_av = np.mean(np.delete(zero_col, zero_max_ind))
        else:
            zero_av = 0.0
        zero_val = np.mean(zero_col[zero_max_ind]) - zero_av
        one_col = one_in[:, i]
        one_max_ind = np.argpartition(one_col, -step)[-step:]
        if rows != step:
            one_av = np.mean(np.delete(one_col, one_max_ind))
        else:
            one_av = 0.0
        one_val = np.mean(one_col[one_max_ind]) - one_av
        max_col_mean.append((one_val + zero_val) / 2)

    max_col_mean = np.array(max_col_mean)
    max_col = np.argmax(max_col_mean).item(0)
    zero_max_ind = np.argpartition(zero_in[:, max_col], -step)[-step:]
    one_max_ind = np.argpartition(one_in[:, max_col], -step)[-step:]
    return max_col, zero_max_ind, one_max_ind


def getDist(a: np.ndarray, b: np.ndarray) -> float:
    c = a - b
    c = np.abs(c)
    c = np.amax(c)
    return c + 0.00001


def getColDistMatrix(zero_in: np.ndarray, one_in: np.ndarray) -> [[float]]:
    data = np.concatenate((zero_in, one_in), axis=0)

    dists = [[0] * (target_features + 1) for i in range(target_features + 1)]
    for i in range(target_features + 1):
        for j in range(i + 1, target_features + 1):
            if i == 0:
                dists[i][j] = dists[j][i] = 1
            else:
                dists[i][j] = dists[j][i] = getDist(data[:, i - 1], data[:, j - 1])
    return dists


def getRowDistMatrix(data_in: np.ndarray) -> [[float]]:
    dists = [[0] * (target_instances + 1) for i in range(target_instances + 1)]
    for i in range(target_instances + 1):
        for j in range(i + 1, target_instances + 1):
            if i == 0:
                dists[i][j] = dists[j][i] = 1
            else:
                dists[i][j] = dists[j][i] = getDist(data_in[i - 1], data_in[j - 1])
    return dists


def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner

    import itertools
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2 ** n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return np.array(list(reversed(path)))


def sortData(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    zero_out = np.copy(zero_in)
    one_out = np.copy(one_in)
    # sort cols
    matrix = getColDistMatrix(zero_in, one_in)
    best_state = held_karp(matrix)
    zero_ind = -np.where(best_state == 0)[0]
    best_state = np.roll(best_state, zero_ind)
    best_state = np.delete(best_state, 0) - 1
    zero_out = zero_out[:, best_state]
    one_out = one_out[:, best_state]
    # sort zero rows
    matrix = getRowDistMatrix(zero_out)
    best_state = held_karp(matrix)
    zero_ind = -np.where(best_state == 0)[0]
    best_state = np.roll(best_state, zero_ind)
    best_state = np.delete(best_state, 0) - 1
    zero_out = zero_out[best_state]

    # sort one rows
    matrix = getRowDistMatrix(one_out)
    best_state = held_karp(matrix)
    zero_ind = -np.where(best_state == 0)[0]
    best_state = np.roll(best_state, zero_ind)
    best_state = np.delete(best_state, 0) - 1
    one_out = one_out[best_state]

    return zero_out, one_out


# def prepareData(data_in: np.ndarray, labels_in: np.ndarray) -> (np.ndarray, np.ndarray):
#     data_in = data_in.astype(np.float64)
#     labels_in = labels_in.astype(np.float64)
#     min_max_scaler = MinMaxScaler()
#     data_out = min_max_scaler.fit_transform(data_in)
#     min_max_scaler = MinMaxScaler()
#     labels_out = min_max_scaler.fit_transform(labels_in)
#     return data_out, labels_out


def writeData(name_in: str, zero_num: int, one_num: int, zero_in: np.ndarray, one_in: np.ndarray) -> bool:
    zero_sorted, one_sorted = sortData(zero_in, one_in)
    raw_path = f'{done_data_str_raw}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_sorted)
    np.save(one_raw_name, one_sorted)
    data_out = np.stack((zero_sorted, one_sorted))
    data_name = f'{done_data_str}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)
    return True


def prepare(dataset_name: str, dataset_in: (np.ndarray, np.ndarray), pbar: tqdm, num: int):
    data_in, classes_in = dataset_in
    labels_in = prepareLabels(classes_in)
    (_, classes_count) = labels_in.shape
    (instances_size, features_size) = data_in.shape
    pbar.set_description("Processing %s/%d:[%d, %d, %d]" % (dirname, num, instances_size, features_size, classes_count))

    if features_size < target_features:
        return False

    per_class_data = [[] for x in range(classes_count)]

    for k in range(classes_count):
        for j in range(instances_size):
            if labels_in[j][k] == 1.0:
                per_class_data[k].append(data_in[j])

    for j in range(classes_count - 1):
        for k in range(j + 1, classes_count):
            zero_data = per_class_data[j]
            zero_len = len(zero_data)
            one_data = per_class_data[k]
            one_len = len(one_data)
            if (zero_len >= target_instances) and (one_len >= target_instances):
                zero_data = np.array(zero_data)
                zero_data = np.array(zero_data[np.random.choice(zero_len, target_instances, replace=False)])
                one_data = np.array(one_data)
                one_data = np.array(one_data[np.random.choice(one_len, target_instances, replace=False)])
                scaled_zero_data, scaled_one_data = scaleData(zero_data, one_data)
                if features_size > target_features:
                    shrunk_zero_data, shrunk_one_data = shrinkFeatures(scaled_zero_data, scaled_one_data)
                else:
                    shrunk_zero_data = scaled_zero_data
                    shrunk_one_data = scaled_one_data
                writeData(dataset_name, j, k, shrunk_zero_data, shrunk_one_data)

    # if instances_size < target_size:
    #     choices = np.random.choice(instances_size, target_size, replace=True)
    #     data_in = data_in[choices]
    #     labels_in = labels_in[choices]
    #     instances_size = target_size
    #
    # labels_out = labels_in
    # if features_size + labels_size < target_size:
    #     choices = np.random.choice(instances_size, target_size, replace=False)
    #     data_out = data_in[choices]
    #     labels_out = labels_in[choices]
    #     instances_size = target_size
    #     try:
    #         data_out = makeFeatures(labels_size, data_out)
    #     except:
    #         return False
    # elif features_size + labels_size == target_size:
    #     data_out = data_in
    # else:
    #     data_out = shrinkFeatures(labels_size, data_in, labels_in)
    #
    # choices = np.random.choice(instances_size, target_size, replace=False)
    # data_to_write = data_out[choices]
    # labels_to_write = labels_out[choices]
    # result = writeData(dataset_name, data_to_write, labels_to_write)
    # return result


if __name__ == '__main__':
    target_features = 16
    target_instances = 64
    target_classes = 2
    raw_data_str = "./datasets/np_raw/"
    done_data_str = f"./datasets/sprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw = f"./datasets/sprocessed_{target_features}_{target_instances}_{target_classes}/raw"
    done_data_str_path = Path(f'{done_data_str}')
    done_data_str_path.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path = Path(f'{done_data_str_raw}')
    done_data_str_raw_path.mkdir(parents=True, exist_ok=True)

    start_from = 0

    non_processed = []
    for (dirpath, dirnames, filenames) in walk(raw_data_str):
        pbar = tqdm(enumerate(dirnames), total=len(dirnames))
        for cc, dirname in pbar:
            if cc >= start_from:
                prepare_status = prepare(dirname, load(dirname), pbar, cc)
                if not prepare_status:
                    non_processed.append(dirname)

    print(non_processed)
