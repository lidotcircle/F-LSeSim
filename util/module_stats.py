import torch


def computeModelGradientsNorm1(model: torch.nn.Module):
    grad_norm = 0
    nparameters = 0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            grad_norm += torch.norm(grad, 1)
            nparameters += grad.numel()
    return grad_norm, nparameters

def computeModelParametersNorm1(model: torch.nn.Module):
    parameters_norm = 0
    nparameters = 0
    for param in model.parameters():
        paramd = param.detach()
        parameters_norm += torch.norm(paramd, 1)
        nparameters += paramd.numel()
    return parameters_norm, nparameters