import time
import pickle
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
eps = 1e-15

########## Data Preprocessing ##########

def read_data(path):
    X, R = [], []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split()
            X.append(list(map(float, tokens[:-1])))
            R.append(int(tokens[-1]))
    data = [cp.asarray(X), cp.asarray(R)]

    return data

def split_data(data, ratio):
    split = int(len(data[0]) * ratio)
    data_tr = [data[i][:split] for i in range(2)]
    data_val = [data[i][split:] for i in range(2)]

    return data_tr, data_val

def window_and_shuffle(data, windowSz):
    N = len(data[0])
    cnt = [0, 0]

    for i in range(windowSz):
        cnt[data[1][i].item()] += 1

    X, R = [], []
    l, r = 0, windowSz
    while True:
        if cnt[0] * cnt[1] == 0:
            X.append(data[0][l:r].flatten())
            R.append(data[1][l])

        if r == N: break
        cnt[data[1][l].item()] -= 1
        cnt[data[1][r].item()] += 1
        l += 1
        r += 1

    X = cp.asarray(X)
    R = cp.asarray(R)
    idx = cp.random.permutation(len(X))

    return [X[idx], R[idx]]


########## Deep Learning Algorithm ##########

def DL(data, E, η, nₕ, L, m):
    nₓ = len(data[0][0])
    nᵧ = 1
    nₛ = len(data[0])

    W = init_W(nₓ, nₕ, nᵧ, L)
    B = init_B(nₓ, nₕ, nᵧ, L)
    ΔW = init_W(nₓ, nₕ, nᵧ, L, zeroInit=True)
    ΔB = init_B(nₓ, nₕ, nᵧ, L)

    loss_prev = float("inf")
    start_time = time.time()
    for epoch in range(E):
        loss_epoch = 0.0
        Batches = create_minibatch(data, m)
        for X, R in Batches:
            batch_size = X.shape[0]

            # forward propagation
            Z = [X] + [None] * (L - 1)
            for l in range(1, L):
                Z[l] = sigmoid(Z[l-1] @ W[l-1].T + B[l-1])

            # back propagation
            loss_epoch += bce_loss(R, Z[L-1]) * batch_size

            for dW in ΔW: dW.fill(0)
            for dB in ΔB: dB.fill(0)
            δ = [None] * (L - 1) + [R - Z[L-1]]

            for l in range(L-1, 0, -1):
                ΔW[l-1] += (δ[l].T @ Z[l-1]) / batch_size
                ΔB[l-1] += δ[l].sum(axis=0) / batch_size
                if 1 < l:
                    δ[l-1] = (δ[l] @ W[l-1]) * Z[l-1] * (1 - Z[l-1])

            for l in range(L - 1):
                W[l] += η * ΔW[l]
                B[l] += η * ΔB[l]

        loss_epoch /= nₛ
        remain_time = (time.time() - start_time) / (epoch + 1) * (E - epoch - 1)

        # if convergence(loss_epoch, loss_prev):
        #     print(f"\nconverged at epoch {epoch+1} with loss = {loss_epoch:.8f}")
        #     break

        print(
            f"\repoch {epoch+1}/{E}\t"
            f"| loss = {loss_epoch:.8f}\t"
            f"| Δloss = {loss_prev - loss_epoch:.8f}\t"
            f"| ETA = {format_time(remain_time)}", end="", flush=True
        )
        loss_prev = loss_epoch

    print("\r", end="", flush=True)

    return W, B


def init_W(nₓ, nₕ, nᵧ, L, zeroInit=False):
    def xavier_init(n_in, n_out):
        limit = cp.sqrt(6 / (n_in + n_out))
        return cp.random.uniform(-limit, limit, size=(n_out, n_in))

    layer_sizes = [nₓ] + [nₕ] * (L - 2) + [nᵧ]
    W = [
        (cp.zeros((n_out, n_in)) if zeroInit else xavier_init(n_in, n_out))
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]

    return W


def init_B(nₓ, nₕ, nᵧ, L):
    layer_sizes = [nₕ] * (L - 2) + [nᵧ]
    bias = [cp.zeros(layer_sizes[i]) for i in range(L - 1)]

    return bias


def create_minibatch(data, batch_size):
    n = len(data[0])
    idx = cp.random.permutation(n)
    batches = []
    for start in range(0, n, batch_size):
        batch_indices = idx[start:start+batch_size]
        x = data[0][batch_indices]
        r = data[1][batch_indices]
        batches.append((x, r[:, None]))

    return batches


def sigmoid(x):
    x = cp.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + cp.exp(-x))


def bce_loss(y, y_hat):
    y_hat = cp.clip(y_hat, eps, 1 - eps)
    return -cp.mean(y * cp.log(y_hat) + (1 - y) * cp.log(1 - y_hat))


def convergence(L, Lprev, rtol=1e-8, atol=1e-10):
    if np.isfinite(Lprev):
        improve_abs = abs(L - Lprev)
        improve_rel = abs(improve_abs / (abs(Lprev) + 1e-12))
        if improve_abs < atol or improve_rel < rtol:
            return True

    return False

def format_time(seconds: float):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m = seconds // 60
        s = seconds % 60
        return f"{m}m {s}s"

    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}h {m}m {s}s"


def test_model(data, W, B):
    X, R = data
    if len(X) == 0:
        return 0, 0
    R = R.ravel()
    L = len(W) + 1

    Z = X
    for l in range(1, L):
        Z = sigmoid(Z @ W[l-1].T + B[l-1])
    y_prob = Z[:, 0]
    y_pred = (y_prob > 0.5).astype(cp.int32)

    tp = int(cp.sum((y_pred == 1) & (R == 1)))
    tn = int(cp.sum((y_pred == 0) & (R == 0)))
    fp = int(cp.sum((y_pred == 1) & (R == 0)))
    fn = int(cp.sum((y_pred == 0) & (R == 1)))

    total = tp + tn + fp + fn
    if total == 0:
        return 0.0, 0.0

    accuracy  = (tp + tn) / total
    err_rate  = 1 - accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0.0

    print(f"tp : {tp}")
    print(f"tn : {tn}")
    print(f"fp : {fp}")
    print(f"fn : {fn}")
    print(f"accuracy : {accuracy}")
    print(f"err_rate : {err_rate}")
    print(f"precision : {precision}")
    print(f"recall : {recall}")
    print(f"f1_score : {f1_score}")

    return err_rate, f1_score

"""# **Optimizing Mini-Batch Size**"""

def find_mini_batch():
    E = 500
    η = 0.01
    w = 5       # window size
    nₕ = 64
    L = 3

    print("preprocessing data...")
    data = read_data('train.txt')
    data_tr, data_val = split_data(data, 0.8)
    data_tr = window_and_shuffle(data_tr, w)
    data_val = window_and_shuffle(data_val, w)

    print("optimizing mini-batch size...")
    err_tr_all, err_val_all = [], []
    err_val_min = float('inf')
    m_candi = [2**i for i in range(6, 17)]

    for m in m_candi:
        print(f"m = {m}")

        W, B = DL(data_tr, E, η, nₕ, L, m)
        err_tr, _ = test_model(data_tr, W, B)
        err_val, _ = test_model(data_val, W, B)

        err_tr_all.append(err_tr)
        err_val_all.append(err_val)

        if err_val_min > err_val:
            err_val_min = err_val
            m_opt = m

    print(f"========== OUTPUT ==========")
    print(f"optimal m : {m_opt}")

    width = 0.35
    x = np.arange(len(m_candi))
    plt.title(f"m_opt = {m_opt}\nerr rate = {err_val_min}")
    plt.figtext(0.5, 0.0, f"E={E}, η={η}, w={w}, nₕ={nₕ}, L={L}, m={m_opt}", ha="center", fontsize=9, color="black")
    plt.bar(x - width/2, err_tr_all, width, color='gray', label='training')
    plt.bar(x + width/2, err_val_all, width, color='red', label='validation')
    plt.xticks(x, m_candi)
    plt.xlabel("mini-batch size")
    plt.ylabel("error rate")
    plt.legend()
    plt.savefig(f"graph_m_{E}_{η}_{w}_{nₕ}_{L}_{m_opt}.png")

    return m_opt

"""# **Optimizing Epoch**"""

def DL_find_epoch(data_tr, data_val, E, η, nₕ, L, m):
    nₓ = len(data_tr[0][0])
    nᵧ = 1
    nₛ = len(data_tr[0])

    W = init_W(nₓ, nₕ, nᵧ, L)
    B = init_B(nₓ, nₕ, nᵧ, L)
    ΔW = init_W(nₓ, nₕ, nᵧ, L, zeroInit=True)
    ΔB = init_B(nₓ, nₕ, nᵧ, L)

    err_tr_list, err_val_list = [], []
    err_min = float('inf')

    loss_prev = float("inf")
    start_time = time.time()
    for epoch in range(E):
        loss_epoch = 0.0
        Batches = create_minibatch(data_tr, m)
        for X, R in Batches:
            batch_size = X.shape[0]

            # forward propagation
            Z = [X] + [None] * (L - 1)
            for l in range(1, L):
                Z[l] = sigmoid(Z[l-1] @ W[l-1].T + B[l-1])

            # back propagation
            loss_epoch += bce_loss(R, Z[L-1]) * batch_size

            for dW in ΔW: dW.fill(0)
            for dB in ΔB: dB.fill(0)
            δ = [None] * (L - 1) + [R - Z[L-1]]

            for l in range(L-1, 0, -1):
                ΔW[l-1] += (δ[l].T @ Z[l-1]) / batch_size
                ΔB[l-1] += δ[l].sum(axis=0) / batch_size
                if 1 < l:
                    δ[l-1] = (δ[l] @ W[l-1]) * Z[l-1] * (1 - Z[l-1])

            for l in range(L - 1):
                W[l] += η * ΔW[l]
                B[l] += η * ΔB[l]

        loss_epoch /= nₛ
        remain_time = (time.time() - start_time) / (epoch + 1) * (E - epoch - 1)

        # if convergence(loss_epoch, loss_prev):
        #     print(f"\nconverged at epoch {epoch+1} with loss = {loss_epoch:.8f}")
        #     break

        print(
            f"\repoch {epoch+1}/{E}\t"
            f"| loss = {loss_epoch}\t"
            f"| Δloss = {loss_prev - loss_epoch:.8f}\t"
            f"| ETA = {format_time(remain_time)}", end="", flush=True
        )
        loss_prev = loss_epoch

        err_tr, _ = test_model(data_tr, W, B)
        err_val, _ = test_model(data_val, W, B)
        err_tr_list.append(err_tr)
        err_val_list.append(err_val)

        if(err_min > err_val):
            err_min = err_val
            W_opt = [w.copy() for w in W]
            B_opt = [b.copy() for b in B]
            E_opt = epoch + 1

    print("\r", end="", flush=True)

    return W_opt, B_opt, E_opt, err_tr_list, err_val_list

def find_epoch():
    E_max = 3000
    η = 0.01
    w = 5
    nₕ = 64
    L = 3
    m_opt = 256

    print("preprocessing data...")
    data = read_data('train.txt')
    data_tr, data_val = split_data(data, 0.8)
    data_tr = window_and_shuffle(data_tr, w)
    data_val = window_and_shuffle(data_val, w)

    print("optimizing epoch...")
    W_opt, B_opt, E_opt, err_tr, err_val = DL_find_epoch(data_tr, data_val, E_max, η, nₕ, L, m_opt)

    print(f"========== OUTPUT ==========")
    print(f"optimal epoch : {E_opt}")
    print(f"error rate : {min(err_val)}")

    plt.title(f"E_opt = {E_opt}\nerr rate = {min(err_val)}")
    plt.figtext(0.5, 0.0, f"E={E_opt}, η={η}, w={w}, nₕ={nₕ}, L={L}, m={m_opt}", ha="center", fontsize=9, color="black")
    plt.plot(err_tr, color='gray', label='training')
    plt.plot(err_val, color='red', label='validation')
    plt.xlabel("epoch")
    plt.ylabel("error rate")
    plt.savefig(f"graph_E_{E_opt}_{η}_{w}_{nₕ}_{L}_{m_opt}.png")

    return E_opt

"""# **Optimizing Learning Rate**"""

def find_learning_rate():
    E_opt = 450
    # E_opt = 100
    w = 5       # window size
    nₕ = 64
    L = 3
    m_opt = 256

    print("preprocessing data...")
    data = read_data('train.txt')
    data_tr, data_val = split_data(data, 0.8)
    data_tr = window_and_shuffle(data_tr, w)
    data_val = window_and_shuffle(data_val, w)

    print("optimizing learning rate...")
    err_tr_all, err_val_all = [], []
    err_min = float('inf')
    η_candi = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # η_candi = [0.05, 0.03, 0.01, 0.007, 0.005, 0.003]

    for η in η_candi:
        print(f"η = {η}")
        W, B = DL(data_tr, E_opt, η, nₕ, L, m_opt)
        err_tr, _ = test_model(data_tr, W, B)
        err_val, _ = test_model(data_val, W, B)

        err_tr_all.append(err_tr)
        err_val_all.append(err_val)

        if err_min > err_val:
            err_min = err_val
            η_opt = η

    print(f"========== OUTPUT ==========")
    print(f"optimal η : {η_opt}")
    print(f"error rate : {err_min}")

    width = 0.35
    x = np.arange(len(η_candi))
    plt.title(f"η_opt = {η_opt}\nerr rate = {err_min}")
    plt.figtext(0.5, 0.0, f"E={E_opt}, w={w}, nₕ={nₕ}, L={L}, m={m_opt}", ha="center", fontsize=9, color="black")
    plt.bar(x - width/2, err_tr_all, width, color='gray', label='training')
    plt.bar(x + width/2, err_val_all, width, color='red', label='validation')
    plt.xticks(x, η_candi)
    plt.xlabel("η")
    plt.ylabel("error rate")
    plt.legend()
    plt.savefig(f"graph_η_{E_opt}_{η_opt}_{w}_{nₕ}_{L}_{m_opt}.png")

    return η_opt

"""# **Optimizing Input Window Size**
기존 시도 : 전체 데이터에 대해 windowing 후 shuffle -> training/validation 나눔

문제점 : training과 validation data의 샘플이 동일한 데이터를 공유함. 실험시 window size가 커질 수록 validation data의 error rate가 단조 감소해 결국 error rate가 0이 되는 문제 발생

해결방안 : K-fold 방법을 사용해 data를 미리 k등분하여 각각의 fold에 대해 windowing 진행.

문제점 : 모델을 평가할 때 주어지는 데이터의 라벨을 데이터는 모르는 상황이어야 하는데, window_and_shuffle 함수를 실행하는 과정에서 이미 데이터의 라벨 정보를 사용하여 데이터를 처리함

해결방안 : evaluation을 위한 데이터를 windowing할 때는 모든 window에 대해 샘플을 추가하되, 라벨은 가운데 샘플의 라벨을 따른다.
"""

def find_max_window_size(data):
    X, labels = data
    n = labels.size
    if n == 0:
        return 0

    diff = cp.diff(labels)
    boundaries = cp.where(diff != 0)[0]

    if boundaries.size == 0:
        return int(n)

    segment_starts = cp.concatenate([cp.array([0]), boundaries + 1])
    segment_ends   = cp.concatenate([boundaries, cp.array([n - 1])])

    lengths = segment_ends - segment_starts + 1
    return int(lengths.min())

def find_window_size():
    E_opt = 450
    η_opt = 0.01
    nₕ = 64
    L = 3
    m_opt = 256

    print("preprocessing data...")
    data = read_data('train.txt')
    data_tr, data_val = split_data(data, 0.8)

    window_max = min(find_max_window_size(data_tr), find_max_window_size(data_val))
    print(f"window max : {window_max}")

    print("optimizing window size...")
    err_tr_all, err_val_all = [], []
    err_min = float('inf')
    w_candi = [i for i in range(1, window_max+1, 2)]

    cp.random.seed(123)
    for w in w_candi:
        print(f"w = {w}")

        data_tr_win = window_and_shuffle(data_tr, w)
        data_val_win = window_and_shuffle(data_val, w)

        W, B = DL(data_tr_win, E_opt, η_opt, nₕ, L, m_opt)
        err_tr, _ = test_model(data_tr_win, W, B)
        err_val, _ = test_model(data_val_win, W, B)

        err_tr_all.append(err_tr)
        err_val_all.append(err_val)

        if err_min > err_val:
            err_min = err_val
            w_opt = w

    print(f"========== OUTPUT ==========")
    print(f"optimal w : {w_opt}")
    print(f"error rate : {err_min}")

    width = 0.35
    x = np.arange(len(w_candi))
    plt.title(f"w_opt = {w_opt}\nerr rate = {err_min}")
    plt.figtext(0.5, 0.0, f"E={E_opt}, w={w_opt}, nₕ={nₕ}, L={L}, m={m_opt}", ha="center", fontsize=9, color="black")
    plt.bar(x - width/2, err_tr_all, width, color='gray', label='training')
    plt.bar(x + width/2, err_val_all, width, color='red', label='validation')
    plt.xticks(x, w_candi)
    plt.xlabel("window size")
    plt.ylabel("error rate")
    plt.legend()
    plt.savefig(f"graph_w_{E_opt}_{η_opt}_{w_opt}_{nₕ}_{L}_{m_opt}.png")

"""# **Optimizing Numbers of Nodes in a Layer**"""

def find_hidden_node_number():
    E_opt = 450
    η_opt = 0.01
    L = 3
    m_opt = 256
    w_opt = 25

    print("preprocessing data...")
    data = read_data('train.txt')
    data_tr, data_val = split_data(data, 0.8)
    data_tr = window_and_shuffle(data_tr, w_opt)
    data_val = window_and_shuffle(data_val, w_opt)

    print("optimizing number of hidden node...")
    err_tr_all, err_val_all = [], []
    err_min = float('inf')
    nₕ_candi = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    cp.random.seed(1218)

    for nₕ in nₕ_candi:
        print(f"nₕ = {nₕ}")

        W, B = DL(data_tr, E_opt, η_opt, nₕ, L, m_opt)
        err_tr, _ = test_model(data_tr, W, B)
        err_val, _ = test_model(data_val, W, B)

        err_tr_all.append(err_tr)
        err_val_all.append(err_val)

        if err_min > err_val:
            err_min = err_val
            nₕ_opt = nₕ

        del W, B
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    print(f"========== OUTPUT ==========")
    print(f"optimal nₕ : {nₕ_opt}")
    print(f"error rate : {err_min}")

    width = 0.35
    x = np.arange(len(nₕ_candi))
    plt.title(f"nₕ_opt = {nₕ_opt}\nerr rate = {err_min}")
    plt.figtext(0.5, 0.0, f"E={E_opt}, η={η_opt}, w={w_opt}, nₕ={nₕ_opt}, L={L}, m={m_opt}", ha="center", fontsize=9, color="black")
    plt.bar(x - width/2, err_tr_all, width, color='gray', label='training')
    plt.bar(x + width/2, err_val_all, width, color='red', label='validation')
    plt.xticks(x, nₕ_candi, rotation=45)
    plt.xlabel("#(hidden node)")
    plt.ylabel("error rate")
    plt.legend()
    plt.savefig(f"graph_nₕ_{E_opt}_{η_opt}_{w_opt}_{nₕ_opt}_{L}_{m_opt}.png")

"""# **Optimizing Number of Layers**"""

def find_layer():
    E_opt = 450
    η_opt = 0.01
    nₕ_opt = 2048
    m_opt = 256
    w_opt = 25

    print("preprocessing data...")
    data = read_data('train.txt')
    data_tr, data_val = split_data(data, 0.8)
    data_tr = window_and_shuffle(data_tr, w_opt)
    data_val = window_and_shuffle(data_val, w_opt)

    print("optimizing number of layers...")
    err_tr_all, err_val_all = [], []
    err_val_min = float('inf')
    L_candi = list(range(2, 10))

    for L in L_candi:
        print(f"L = {L}")

        W, B = DL(data_tr, E_opt, η_opt, nₕ_opt, L, m_opt)
        err_tr, _ = test_model(data_tr, W, B)
        err_val, _ = test_model(data_val, W, B)

        err_tr_all.append(err_tr)
        err_val_all.append(err_val)

        if err_val_min > err_val:
            err_val_min = err_val
            L_opt = L

    print(f"========== OUTPUT ==========")
    print(f"optimal L : {L_opt}")

    width = 0.35
    x = np.arange(len(L_candi))
    plt.title(f"L_opt = {L_opt}\nerr rate = {err_val_min}")
    plt.figtext(0.5, 0.01, f"E={E_opt}, η={η_opt}, w={w_opt}, nₕ={nₕ_opt}, L={L_opt}, m={m_opt}", ha="center", fontsize=9, color="black")
    plt.bar(x - width/2, err_tr_all, width, color='gray', label='training')
    plt.bar(x + width/2, err_val_all, width, color='red', label='validation')
    plt.xticks(x, L_candi)
    plt.xlabel("#(layer)")
    plt.ylabel("error rate")
    plt.legend()
    plt.savefig(f"graph_L_{E_opt}_{η_opt}_{w_opt}_{nₕ_opt}_{L_opt}_{m_opt}.png")

"""# **Test Model**"""

def final_test(seed=None):
    if seed is not None:
        cp.random.seed(seed)

    E = 450
    η = 0.01
    nₕ = 2048
    L = 7
    m = 256
    w = 25

    print("preprocessing data...")
    data = read_data("train.txt")
    data = window_and_shuffle(data, w)
    data_test = read_data("test.txt")
    data_test = window_and_shuffle(data_test, w)

    print("training model...")
    W, B = DL(data, E, η, nₕ, L, m)

    print("===== Train Data Result =====")
    err_rate, f1_score = test_model(data, W, B)
    print()

    print("===== Test Data Result =====")
    err_rate, f1_score = test_model(data_test, W, B)

    # print(f"========== OUTPUT ==========")
    # print(f"error rate : {err_rate}")
    # print(f"f1 score : {f1_score}")

    with open(f"model.pkl", "wb") as f:
        pickle.dump((W,B), f)

    return err_rate

def test_with_model():
    with open(f"model.pkl", "rb") as f:
        W, B = pickle.load(f)

    data = read_data("train.txt")
    data = window_and_shuffle(data, 25)

    err, f1 = test_model(data, W, B)

if __name__ == "__main__":
    err_rate = []
    for i in [123, 124, 125, 126, 127]:
        print(f"seed = {i}")
        err = final_test(i)
        err_rate.append(err)
    print("===== OUTPUT =====")
    print(f"mean error rate : {np.mean(err_rate)}")
    print(f"error rate : {err_rate}")