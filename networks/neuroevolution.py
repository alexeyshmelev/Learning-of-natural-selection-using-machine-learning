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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii.evaluator.pytorch.lightning import LightningModule


@model_wrapper
class ModelSpace(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        filter_size = nn.ValueChoice([3, 5, 7])
        num_channels = nn.ValueChoice([64, 128, 256])

        self.conv1 = nn.LayerChoice([
            nn.Conv1d(501, num_channels, filter_size),
            nn.Conv1d(501, num_channels, filter_size),
            nn.Conv1d(501, num_channels, filter_size)
        ])

        self.batch1 = nn.BatchNorm1d(num_channels)

        self.dense

    def forward(self, x):

        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        pred = self.forward(x)  # model is the one that is searched for
        loss = self.loss_fn(pred, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)  # model is the one that is searched for
        loss = self.loss_fn(pred, y)
        # Logging to TensorBoard by default
        # self.log('vall_loss', loss)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_validation_epoch_end(self):
        print('ok1')
        # nni.report_intermediate_result(self.trainer.callback_metrics['val_loss'].item())

    def teardown(self, stage):
        print('ok2')
        # if stage == 'fit':
        #     nni.report_final_result(self.trainer.callback_metrics['val_loss'].item())


model_space = ModelSpace()

train_dataset = TensorDataset(torch.tensor(np.load(r'C:\HSE\EPISTASIS\nn\all_inputs_testdd.npy')).to(device='cuda', dtype=torch.float), torch.tensor(np.argmax(np.load(r'C:\HSE\EPISTASIS\nn\all_targets_testdd.npy').squeeze(), axis=1)).to(device='cuda'))
test_dataset = TensorDataset(torch.tensor(np.load(r'C:\HSE\EPISTASIS\nn\all_inputs_testdd.npy')).to(device='cuda', dtype=torch.float), torch.tensor(np.argmax(np.load(r'C:\HSE\EPISTASIS\nn\all_targets_testdd.npy').squeeze(), axis=1)).to(device='cuda'))

# evaluator = pl.Classification(
#     train_dataloaders=pl.DataLoader(train_dataset, batch_size=32, shuffle=True),
#     val_dataloaders=pl.DataLoader(test_dataset, batch_size=32, shuffle=True),
#     optimizer=torch.optim.Adam,
#     gpus=1,
#     max_epochs=1
# )

lightning = pl.Lightning(ModelSpace(),
                         pl.Trainer(max_epochs=1, accelerator='gpu', devices=1, check_val_every_n_epoch=1),
                         train_dataloaders=pl.DataLoader(train_dataset, batch_size=32, shuffle=True),
                         val_dataloaders=pl.DataLoader(test_dataset, batch_size=32, shuffle=True))

exploration_strategy = strategy.DARTS() # (just reminder) Evaluator needs to be a lightning evaluator to make one-shot strategy work.

exp = RetiariiExperiment(model_space, lightning, [], exploration_strategy) # base_model, evaluator, mutators, strategy
exp_config = RetiariiExeConfig()

exp_config.execution_engine = 'oneshot'

exp.run(exp_config, 8081)

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)
    exported_model = model_dict

# with nni.retiarii.utils.original_state_dict_hooks(ModelSpace):
#     supernet_style_state_dict = ModelSpace.state_dict()

with nni.retiarii.fixed_arch(exported_model):
    model = ModelSpace()
# Then use evaluator.evaluate
dummy_input = torch.zeros(1, 501, 9)
torch.onnx.export(model, (dummy_input, ), 'model_troch_export.onnx', training=torch.onnx.TrainingMode.TRAINING,  # export in Training mode
                      verbose=True,
                      export_params=True)
lightning.evaluate(model)

print(model)