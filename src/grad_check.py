import torch
from prepare import prepare_data
from model import ECGNet, cross_entropy
if __name__ == '__main__':
    X_train, y_train, *_ = prepare_data()
    torch.manual_seed(0)
    model = ECGNet()
    xb, yb = X_train[:64], y_train[:64]
    logits = model(xb)
    loss = cross_entropy(logits, yb)
    loss.backward()
    with torch.no_grad():
        Y_hat = torch.softmax(logits, dim=1)
        Y = torch.zeros_like(logits)
        Y[torch.arange(64), yb] = 1.0
        gW3 = ((Y_hat - Y).T @ model.last_A2) / 64.0
        gb3 = (Y_hat - Y).mean(dim=0)
        print('Max diff W3:', (gW3 - model.W3.grad).abs().max().item())
        print('Max diff b3:', (gb3 - model.b3.grad).abs().max().item())
