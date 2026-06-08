import torch
import torch.nn.functional as F


def compute_class_weights(y, train_mask):
    """
    Calcula pesos de clase usando solo nodos de entrenamiento.
    """
    y_train = y[train_mask]

    num_classes = int(y.max().item()) + 1
    counts = torch.bincount(y_train, minlength=num_classes).float()

    total = counts.sum()
    weights = total / (num_classes * counts)

    return weights


def train_one_epoch(model, data, optimizer, criterion, device):
    model.train()

    data = data.to(device)

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def predict_scores(model, data, device):
    """
    Devuelve probabilidades de clase positiva.
    """
    model.eval()
    data = data.to(device)

    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]

    return probs.cpu()
