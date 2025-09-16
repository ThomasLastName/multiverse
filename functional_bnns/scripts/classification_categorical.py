import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from bnns import GaussianBNN
from bnns.utils.handling import support_for_progress_bars
torch.manual_seed(2025)


#
# ~~~ Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root=os.path.join("..", "bnns", "data"),
    train=True,
    transform=transform,
    download=True,
)
test_dataset = datasets.MNIST(
    root=os.path.join("..", "bnns", "data"),
    train=False,
    transform=transform,
    download=True,
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


#
# ~~~ Define architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
architecture = nn.Sequential(
    nn.Flatten(start_dim=-3),  # flatten last 3 dims: [B,1,28,28] -> [B,784] and [n,B,1,28,28] -> [n,B,784]
    nn.Linear(784, 300),
    nn.ReLU(),
    nn.Linear(300, 300),
    nn.ReLU(),
    nn.Linear(300, 300),
    nn.ReLU(),
    nn.Linear(300, 10),  # raw logits (no softmax here!)
)
architecture = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # [B, 32, 28, 28]
    nn.ReLU(),
    nn.MaxPool2d(2),                                                                # [B, 32, 14, 14]
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),                          # [B, 64, 14, 14]
    nn.ReLU(),
    nn.MaxPool2d(2),                                                                # [B, 64, 7, 7]
    nn.Flatten(start_dim=-3),  # flatten last 3 dims: [B,1,28,28] -> [B,784] and [n,B,1,28,28] -> [n,B,784]
    nn.Linear(64 * 7 * 7, 400),
    nn.ReLU(),
    nn.Linear(400, 10)
)

#
# ~~~ Metric
def evaluate_accuracy(model, dataloader, device, n_samples=30):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), support_for_progress_bars():
        for X, y in tqdm( dataloader, total=len(dataloader), desc="Measuring Test Accuracy" ):
            X, y = X.to(device), y.to(device)
            # Draw n_samples weight samples at once
            preds = model(X, n=n_samples)
            mean_preds = preds.mean(0)
            predicted_classes = mean_preds.argmax(dim=1)
            correct += (predicted_classes == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


#
# ~~~ Train model
vi, lik, kl, acc = [], [], [], []
bnn = GaussianBNN(*architecture).to(device)
# bnn = GaussianBNN(*architecture, projection_method="torchbnn").to(device)
optimizer = torch.optim.Adam(bnn.parameters(), lr=0.005)  # start a bit higher
kl_weight = 0.0001

for e in range(4):
    for (X, y) in tqdm(train_loader, desc=f"epoch {e+1}"):
        X, y = X.to(device), y.to(device)   # y are integer class labels
        bnn.resample_weights()              # draw new weight sample
        yhat = bnn(X)                       # logits: [batch, 10]
        log_lik = torch.distributions.Categorical(logits=yhat).log_prob(y).sum()
        kl_div  = bnn.weight_kl()
        vi_loss = kl_weight*kl_div - log_lik          # negative ELBO
        vi_loss.backward()
        # bnn.apply_chain_rule_for_soft_projection()
        optimizer.step()
        optimizer.zero_grad()
        train_acc = ( yhat.detach().argmax(dim=1) == y ).sum() / len(y)
        # bnn.apply_soft_projection()
        vi.append(vi_loss.item())
        lik.append(log_lik.item())
        kl.append(kl_div.item())
        acc.append(train_acc.item())
    a = evaluate_accuracy(bnn, test_loader, device, n_samples=100)
    print(f"Epoch {e+1}: Test accuracy = {a:.4f}")

