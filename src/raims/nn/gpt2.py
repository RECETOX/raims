from typing import Dict

from transformers import GPT2Config, GPT2LMHeadModel
from pytorch_lightning import LightningModule


class Gpt2(LightningModule):
    def __init__(self, vocabulary: Dict[str, int]):
        super().__init__()

        config = GPT2Config(n_positions=256, vocab_size=len(vocabulary) + 1, bos_token_id=len(vocabulary),
                            eos_token_id=len(vocabulary))
        self.model = GPT2LMHeadModel(config)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        outputs = self.model(**batch)
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
