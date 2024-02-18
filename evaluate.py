import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1, bootstrap_iters=1000):
    all_preds = []
    all_trues = []
    all_test_loss = []
    for _ in range(bootstrap_iters):
        pred = []
        true = []
        test_loss = 0
        for test_batch, label in test_loader:
            test_batch, label = test_batch.to(device), label.to(device)
            batch_len = test_batch.shape[0]
            observed_data, observed_mask, observed_tp = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
            with torch.no_grad():
                out = model(torch.cat((observed_data, observed_mask), 2), observed_tp)
                # 이후 로직은 원래대로 유지하되, bootstrap_iters만큼 반복하여 샘플링된 데이터에 대해 평가
                # ...
        # 각 bootstrap iteration의 예측과 실제 레이블을 저장
        all_preds.append(pred)
        all_trues.append(true)
        all_test_loss.append(test_loss)

    # Bootstrap된 결과로부터 각 iteration의 AUC 계산
    auc_scores = [roc_auc_score(all_trues[i], all_preds[i]) for i in range(bootstrap_iters)]

    # 신뢰구간 계산
    lower_bound = np.percentile(auc_scores, 2.5)
    upper_bound = np.percentile(auc_scores, 97.5)

    # 평균 테스트 손실, 평균 정확도, 평균 AUC 및 AUC의 신뢰구간 반환
    mean_test_loss = np.mean(all_test_loss)
    mean_auc = np.mean(auc_scores)
    print(f"AUC 95% 신뢰구간: {lower_bound} ~ {upper_bound}")
    return mean_test_loss, mean_auc, (lower_bound, upper_bound)