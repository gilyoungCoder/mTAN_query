from physionet import *
import torch
from utils import *
q = 0.016
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, 8000),
                                  device=device)
# Use custom collate_fn to combine samples with arbitrary time observations.


# Combine and shuffle samples from physionet Train and physionet Test
train_data = train_dataset_obj[:len(train_dataset_obj)]



record_id, tt, vals, mask, labels = train_data[0]
print(vals.shape)
# n_samples = len(total_dataset) 특성 숫자/텐서의 마지막 차원
input_dim = vals.size(-1)
data_min, data_max = get_data_min_max(train_data, device)
print(f"data_min", data_min)
print(f"data_max", data_max)
train_data_combined, _ = variable_time_collate_fn(train_data, device, classify=True,
                                                      data_min=data_min, data_max=data_max)

print(train_data_combined.shape)
vals, mask = train_data_combined[:, :, :41], train_data_combined[:, :, 41:82] 
filtered_vals = vals * mask
valid_counts = mask.sum(dim=(0, 1))

feature_means = filtered_vals.sum(dim=(0, 1)) / (valid_counts + 1e-10)

print(feature_means)
torch.save(feature_means, 'data/physionet/PhysioNet/x_means.pt')

# if flag:
#     test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
#                                                     data_min=data_min, data_max=data_max)

#     if args.classif:
#         train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
#                                                                 random_state=11, shuffle=True)
#         train_data_combined = variable_time_collate_fn(
#             train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
#         val_data_combined = variable_time_collate_fn(
#             val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
#         print(train_data_combined[1].sum(
#         ), val_data_combined[1].sum(), test_data_combined[1].sum())
#         print(train_data_combined[0].size(), train_data_combined[1].size(),
#                 val_data_combined[0].size(), val_data_combined[1].size(),
#                 test_data_combined[0].size(), test_data_combined[1].size())

#         train_data_combined = TensorDataset(
#             train_data_combined[0], train_data_combined[1].long().squeeze())
#         val_data_combined = TensorDataset(
#             val_data_combined[0], val_data_combined[1].long().squeeze())
#         test_data_combined = TensorDataset(
#             test_data_combined[0], test_data_combined[1].long().squeeze())
