from transformers import GPT2Config, GPT2LMHeadModel
from pytorch_lightning import LightningModule


class Gpt2Module(LightningModule):
    def __init__(self, vocab):
        super(self).__init__()

        config = GPT2Config(n_positions=256, vocab_size=len(vocab) + 1, bos_token_id=len(vocab), eos_token_id=len(vocab))
        self.model = GPT2LMHeadModel(config)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        outputs = self.model(**batch)
        return outputs.loss

    def training_step(self, batch, batch_ids):
        loss = self._step(batch)
        self.log("train_loss", loss)

    def validation_step(self, batch, batch_ids):
        loss = self._step(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_ids):
        loss = self._step(batch)
        self.log("test_loss", loss, sync_dist=True)
