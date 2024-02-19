import torch

# 예시: delta 텐서 생성
delta = torch.tensor([1.0])

# X 텐서의 크기를 나타내는 예시 (예를 들어, 2x2x2)
X = torch.randn(2, 2, 2)  # X 텐서의 실제 크기와 상관없는 예시 값

# delta 값을 X 텐서와 같은 크기로 반복
# torch.tile(delta, X.shape)는 torch.tile(delta, (2, 2, 2))와 동일하게 작동하지만,
# 아래 코드는 일반적인 사용 예시를 위해 수정됩니다.
repeated_delta = torch.tile(delta, X.size())

print(repeated_delta)
