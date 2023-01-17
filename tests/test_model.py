import torch

from src.models.model import CIFAR10Module, create_model


def test_model_output():
    model = CIFAR10Module(classifier=create_model())

    batchsize = 64
    x = torch.rand((batchsize, 3, 32, 32))
    out = model.forward(x)

    assert out.shape == torch.Size([batchsize, 10])
    assert torch.sum(torch.exp(out)) == batchsize


if __name__ == "__main__":
    test_model_output()
