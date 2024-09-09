import torch 
import torch.nn as nn  
def architecture():
    sequential = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(8704, 128),
        nn.ReLU(), 
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return sequential


def main():
    model = architecture()
    a = torch.rand((3,32,69)).unsqueeze(0)
    print(model(a))
    return

if __name__ == "__main__":
    main()