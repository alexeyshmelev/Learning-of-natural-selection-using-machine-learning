import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy
import nni
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_switch = nn.InputChoice(n_candidates=2)

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.9)
        self.dropout3 = nn.Dropout(0.5)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        # s1 = self.dropout1(x)
        # s2 = self.dropout2(x)
        selected = self.input_switch([self.dropout1(x), self.dropout2(x)])
        x = torch.flatten(selected, 1)
        x = self.fc2(self.dropout3(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


model_space = ModelSpace()

search_strategy = strategy.Random(dedup=True)

def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)

transform = nni.trace(transforms.Compose)([nni.trace(transforms.ToTensor)(), nni.trace(transforms.Normalize)((0.1307,), (0.3081,))])

def create_mnist_dataset(root, transform):
  return MNIST(root='data/mnist', train=False, download=True, transform=transform)

test_dataset = nni.trace(create_mnist_dataset)('data/mnist', transform=transform)  # factory is also acceptable
train_dataset = nni.trace(MNIST)(root='data/mnist', train=True, download=True, transform=transform)

evaluator = pl.Classification(
  # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
  # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
  train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
  val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
  # Other keyword arguments passed to pytorch_lightning.Trainer.
  gpus=1,
  max_epochs = 2,
)

exploration_strategy = strategy.DARTS()

exp = RetiariiExperiment(model_space, evaluator, [], exploration_strategy)
exp_config = RetiariiExeConfig()
# exp_config.experiment_name = 'mnist_search'
#
# exp_config.max_trial_number = 4   # spawn 4 trials at most
# exp_config.trial_concurrency = 2  # will run two trials concurrently
#
# exp_config.trial_gpu_number = 1
# exp_config.training_service.use_active_gpu = True

exp_config.execution_engine = 'oneshot'
export_formatter = 'code'

exp.run(exp_config, 8081)

# for model_dict in exp.export_top_models(formatter='code'):
#     print(model_dict)

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)