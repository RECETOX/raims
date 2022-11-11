from typing import Dict
import torch

from transformers import GPT2Config, GPT2LMHeadModel
from pytorch_lightning import LightningModule


class Gpt2(LightningModule):
    def __init__(self, vocabulary: Dict[str, int],learning_rate = 1e-3,n_embd=768,n_layer=12,n_head=12):
        super().__init__()

        config = GPT2Config(n_positions=256, vocab_size=len(vocabulary) + 1, bos_token_id=len(vocabulary),
                            eos_token_id=len(vocabulary),
                            n_embd=n_embd,n_layer=n_layer,n_head=n_head)
        self.model = GPT2LMHeadModel(config)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        outputs = self.model(**batch)
#        print(outputs)
        return outputs.loss

    def training_step(self, batch, batch_ids):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_ids):
        loss = self._step(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_ids):
        loss = self._step(batch)
        self.log("test_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

