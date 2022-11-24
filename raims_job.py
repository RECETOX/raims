#!/usr/bin/env python3

import argparse
from raims.data import load_word2vec
from raims.data import HuggingfaceDataModule
from raims.nn import Gpt2
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import matchms.logging_functions as mmsl
import torch.multiprocessing as mp


def main():
	mp.set_start_method('spawn', force=True) 
	parser = argparse.ArgumentParser()
	parser.add_argument('--heads',type=int)
	parser.add_argument('--layers',type=int)
	parser.add_argument('--embed',type=int)
	parser.add_argument('--name',type=str,default='noname')
	parser.add_argument('--project',type=str,default='raims')
	parser.add_argument('--gpus',type=int,default=1)
	parser.add_argument('--batch',type=int,default=64)
	parser.add_argument('--half',action='store_true')
	parser.add_argument('--keyfile',type=str,default='wandb/key')
	
	args = parser.parse_args()
	
	tags = [
		f'heads_{args.heads}', 
		f'layers_{args.layers}', 
		f'embed_{args.embed}', 
		f'gpus_{args.gpus}', 
		f'batch_{args.batch}', 
		f'name_{args.name}'
	]
	
	
	if args.half:
		tags.append('half')
		precision = 16
	else:
		precision = 32
	
	with open(args.keyfile) as kf:
		key = kf.readline().rstrip()
	
	name=f'{args.name}_H{args.heads}_L{args.layers}_E{args.embed}_G{args.gpus}_B{args.batch}_P{precision}'
	
	
	mmsl.add_logging_to_file("matchms.log",remove_stream_handlers=True)
	
	wandb.login(key=key)
	wandb.init(project=args.project,name=name,tags=tags)
	logger = WandbLogger(offline=False)
	
	vocabulary, _ = load_word2vec(path='model/mona-random-w2v.model')
	datamodule = HuggingfaceDataModule(path='data/split/mona-random', vocabulary=vocabulary, batch_size=args.batch,num_workers=0)
	
	model = Gpt2(vocabulary=vocabulary, n_embd=args.embed, n_layer=args.layers, n_head=args.heads)
	
	trainer =  Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=3)], logger=logger, max_epochs=500, accelerator='gpu', devices=args.gpus, precision=precision)
	
	trainer.fit(model=model,datamodule=datamodule)

if __name__ == '__main__':
    main()


