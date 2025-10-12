import numpy as np
import matplotlib.pyplot as plt
eps = 1e-8

#============================================================
# EM 학습 실행
#============================================================
def main():
    K = 20
    K_fold = 10
    max_iter = 250

    data, label = read_data('train.txt')
    fold_idx = split_data(len(data), K_fold)

    err_train_mean, err_valid_mean = [], []
    for k in range(1, K + 1):
        print(f"K = {k} 훈련 시작~~~~~~~~~~~~~~~!!!!!")

        err_train, err_valid = [], []
        for k_fold in range(0, K_fold):
            print(f"    {k_fold+1}번째 fold 학습 중...")
            data_tr, label_tr = data[fold_idx[k_fold][0]], label[fold_idx[k_fold][0]]
            data_val, label_val = data[fold_idx[k_fold][1]], label[fold_idx[k_fold][1]]
            pi, mu, Sigma = init_parameters(data_tr, k)
            w = EM(data_tr, k, max_iter, pi, mu, Sigma)
            P_c, pi, mu, Sigma = classifier_parameters(data_tr, label_tr, k, w)

            print(f"    training, validation 데이터 검증 중...")
            err_train.append(test_model(data_tr, label_tr, k, P_c, pi, mu, Sigma))
            err_valid.append(test_model(data_val, label_val, k, P_c, pi, mu, Sigma))

        err_train_mean.append(np.mean(err_train))
        err_valid_mean.append(np.mean(err_valid))

    plt.title('Expectation Maximization')
    plt.plot(range(1, len(err_train_mean) + 1), err_train_mean, color='gray', label='training')
    plt.plot(range(1, len(err_valid_mean) + 1), err_valid_mean, color='red', label='validation')
    plt.show()

    print(f"최적 K 찾고 전체 데이터에 대해 학습중...")
    K_opt = np.argmin(err_valid_mean) + 1
    pi, mu, Sigma = init_parameters(data, K_opt)
    w_opt = EM(data, K_opt, np.inf, pi, mu, Sigma)
    P_c, pi, mu, Sigma = classifier_parameters(data, label, K_opt, w_opt)

    print(f"최적 K로 test 데이터에 대해 최종 평가 중...")
    data_test, label_test = read_data('test.txt')
    error = test_model(data_test, label_test, K_opt, P_c, pi, mu, Sigma)

    print(f"========== 최종 결과 ==========")
    print(f"P(c_k) : \n{P_c}")
    print(f"P(g_{{k_c, k_g}}) : \n{pi}")
    print(f"mu_{{k_c, k_g}} : \n{mu}")
    print(f"Sigma_{{k_c, k_g}} : \n{Sigma}")
    print(f"\n최적 K = {K_opt}, error = {error}")


#============================================================
# EM 알고리즘
#============================================================
def EM(data, K, max_iter, pi, mu, Sigma):
    L = -np.inf
    cnt = 0
    while True:
        cnt += 1
        Lprev = L
        w, L = E_step(data, pi, mu, Sigma)
        pi, mu, Sigma = M_step(data, K, w)
        if convergence(L, Lprev) or cnt >= max_iter: break

    return w

def E_step(X, pi, mu, Sigma):
    N, d = X.shape

    # alpha_k     := -0.5 log |Sigma_k|
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, 1, 2)) + eps * np.eye(d)
    alpha = -0.5 * np.linalg.slogdet(Sigma)[1]

    # beta_{t,k}  := -0.5 (x^t - mu_k)^T Sigma_i^-1 (x^t - mu_k)
    Sigma_inv = np.linalg.inv(Sigma)  # (K,d,d)
    diff = X[:, None, :] - mu[None, :, :]  # (N,K,d)
    xS = np.einsum('nkd,kde->nke', diff, Sigma_inv)
    mahalanobis = np.sum(xS * diff, axis=-1)
    beta = -0.5 * mahalanobis

    # gamma_k     := log pi_k
    pi = np.clip(pi, eps, 1.0)
    pi = pi / pi.sum()
    gamma = np.log(pi)

    # l_{t,k}     := alpha_k + beta_{t,k} + gamma_k
    l = alpha[None, :] + beta + gamma[None, :]

    # sum_l_t     := l_{t,0} + log (exp(l_{t,0}-l_{t,0}) + ... + exp(l_{t,k-1}-l_{t,0}))
    m = np.max(l, axis=1, keepdims=True)
    sum_l = m + np.log(np.sum(np.exp(l - m), axis=1, keepdims=True))  # (N,1)

    # log w_{t,k} := l_{t,k} - sum_l_t
    log_w = l - sum_l
    w = np.exp(log_w)
    L = np.sum(sum_l.squeeze(1))

    return w, L

def M_step(X, K, w):
    N, d = X.shape
    N_k = np.sum(w, axis=0)
    mu = (w.T @ X) / N_k[:, None]
    Sigma = np.zeros((K, d, d))
    for k in range(K):
        diff = X - mu[k]
        Sigma[k] = (diff.T * w[:, k]) @ diff / N_k[k]
    pi = N_k / N

    return pi, mu, Sigma

def convergence(L, Lprev, rtol=1e-4, atol=1e-6):
    if np.isfinite(Lprev):
        improve_abs = abs(L - Lprev)
        improve_rel = abs(improve_abs / (abs(Lprev) + 1e-12))
        if improve_abs < atol or improve_rel < rtol:
            return True

    return False


#============================================================
# 학습 및 테스트 관련 함수
#============================================================
def init_parameters(X, K):
    N, d = X.shape

    Sigma = np.cov(X, rowvar=False) + eps * np.eye(d)
    eigenValues, eigenVectors = np.linalg.eigh(Sigma)
    pc1 = eigenVectors[:, -1]

    proj = X @ pc1  # 행렬곱
    bins = np.percentile(proj, np.linspace(0, 100, K + 1))
    idx = np.digitize(proj, bins[1:-1], right=True)

    pi, mu, Sigma = [], [], []
    for k in range(K):
        group = X[idx == k]
        pi.append(len(group) / N)           # pi[k] = P(g_k | theta)
        mu.append(np.mean(group, axis=0))   # mu[k] = mean vector of g_k
        Sigma.append(np.cov(group, rowvar=False) + eps * np.eye(d))   # Sigma[k] = Cov matrix of g_k

    return map(np.array, (pi, mu, Sigma))

def classifier_parameters(X, r, K, w):
    N, d = X.shape

    # x : (N,d)
    # w : (N,k)
    r = np.stack([1 - r, r], axis=1)  # (N,2)
    N_r = r.sum(axis=0)[:, None]            # (2,1)
    N_rw = r.T @ w                          # (2,K)

    wx = (w[:, :, None] * X[:, None, :]).reshape(N, K * d)
    N_rwx = (r.T @ wx).reshape(2, K, d)  # (2,k,d)

    xx = X[:, :, None] * X[:, None, :]  # (N, d, d)
    wxx = (w[:, :, None, None] * xx[:, None, :, :]).reshape(N, K * d * d)
    N_rwxx = (r.T @ wxx).reshape(2, K, d, d)    # (2,K,d,d)

    P_c = N_r / N       # (2,1)
    pi = N_rw / N_r     # (2,K)
    mu = N_rwx / N_rw[..., None]   # (2,K,d)
    Sigma = N_rwxx / N_rw[..., None, None] - mu[..., :, None] @ mu[..., None, :]   # (2,K,d,d)

    pi = pi / (pi.sum(axis=1, keepdims=True) + eps)
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    Sigma = Sigma + (eps * np.eye(d))[None, None, :, :]

    return P_c, pi, mu, Sigma

def test_model(X, r, K, P_c, pi, mu, Sigma):
    # X(N,d), r(N,), K()
    # P_c(2,1), pi(2,K), mu(2,K,d), Sigma(2,K,d,d)
    N, d = X.shape

    alpha = -0.5 * (d * np.log(2*np.pi) + np.log(np.linalg.det(Sigma)))     # (2,K)
    Sigma_inv = np.linalg.inv(Sigma)    # (2,K,d,d)
    gamma = np.log(pi + eps)  # (2,K)

    def classify(x):
        # x: (d,)
        diff = x[None, None, :] - mu    # (2,K,d)
        # xS = diff @ Sigma_inv  # (2,K,d)
        xS = np.einsum('ckd,ckde->cke', diff, Sigma_inv)
        mahalanobis = np.sum(xS * diff, axis=-1)  # (2,K)
        beta = -0.5 * mahalanobis  # (2,K)
        loglik_ck = alpha + beta + gamma    # (2,K)

        m = np.max(loglik_ck, axis=1)    # (2,1)
        log_px_c = (m + np.log(np.sum(np.exp(loglik_ck - m[:, None]), axis=1)))    # (2,)
        log_post = log_px_c + np.log(P_c.squeeze(-1) + eps) # (2,)

        return int(np.argmax(log_post))

    correct = 0
    for t in range(N):
        if r[t] == classify(X[t]):
            correct += 1
    err = 1.0 - (correct / N)

    return err


#============================================================
# 유틸 함수
#============================================================
def read_data(path, seed=42):
    data_list, label_list = [], []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split()  # 공백 단위로 끊어서 저장
            data_list.append(list(map(float, tokens[:-1])))
            label_list.append(int(tokens[-1]))
    data = np.array([np.array(x) for x in data_list])
    label = np.array(label_list)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))

    return data[idx], label[idx]

def split_data(N, K_fold):
    fold_size = N // K_fold
    fold_idx = []
    for kth in range(1, K_fold+1):
        start = (kth - 1) * fold_size
        end = N if kth == K_fold else kth * fold_size
        indices = np.arange(N)
        train_idx = np.concatenate((indices[:start], indices[end:]))
        valid_idx = indices[start:end]
        fold_idx.append([train_idx, valid_idx])

    return fold_idx


if __name__ == "__main__":
    main()