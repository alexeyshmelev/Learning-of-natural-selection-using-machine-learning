import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy
import nni
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl


@nni.trace
class SampleDataset(Dataset):
    def __init__(self, secuence):
        self.samples = torch.tensor(secuence)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (self.samples[index])


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_switch = nn.InputChoice(n_candidates=2)

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
        ])
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.9)
        self.dropout3 = nn.Dropout(0.5)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        selected = self.input_switch([self.dropout1(x), self.dropout2(x)])
        x = torch.flatten(selected, 1)
        x = self.fc2(self.dropout3(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


model_space = ModelSpace()


# evaluator = pl.Classification(
#     train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
#     val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
#     optimizer=torch.optim.Adamy,
#     gpus=1,
#     max_epochs=2
# )
#
# exploration_strategy = strategy.DARTS()
#
# exp = RetiariiExperiment(model_space, evaluator, [], exploration_strategy)
# exp_config = RetiariiExeConfig()
#
# exp_config.execution_engine = 'oneshot'
#
# exp.run(exp_config, 8081)
#
# for model_dict in exp.export_top_models(formatter='dict'):
#     print(model_dict)