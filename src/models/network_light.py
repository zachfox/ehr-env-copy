import numpy as np
import pickle as pk

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from tqdm import tqdm as bar

from model import Encoder, Decoder
                    


class Senseiver(pl.LightningModule):
    def __init__(self,**kwargs):
    
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        
        
        # pos_encoder_ch = self.hparams.space_bands*len(self.hparams.image_size)*2
        # pos_encoder_ch = 3        
        pos_encoder_ch = self.hparams.space_bands*3*2
        # self.hparams.im_ch = 7
        self.hparams.im_ch = 7+self.hparams.nlags 
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        self.r2_score = R2Score()

        self.encoder = Encoder(
            input_ch = self.hparams.im_ch+pos_encoder_ch,
            preproc_ch = self.hparams.enc_preproc_ch,
            num_latents = self.hparams.num_latents,
            num_latent_channels = self.hparams.enc_num_latent_channels,
            num_layers = self.hparams.num_layers,
            num_cross_attention_heads = self.hparams.num_cross_attention_heads,
            num_self_attention_heads = self.hparams.enc_num_self_attention_heads,
            num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block,
            dropout = self.hparams.dropout,
        )
        
       
        self.decoder_1 = Decoder(
            ff_channels = pos_encoder_ch+self.hparams.im_ch-1-self.hparams.nlags,
            preproc_ch = self.hparams.dec_preproc_ch,  # latent bottleneck
            num_latent_channels = self.hparams.dec_num_latent_channels,  # hyperparam
            latent_size = self.hparams.latent_size,  # collapse from n_sensors to 1
            # num_output_channels = self.hparams.im_ch,
            num_output_channels = 1,
            num_cross_attention_heads = self.hparams.dec_num_cross_attention_heads,
            dropout = self.hparams.dropout,
        )

        
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')
        
        
        
    def forward(self, sensor_values, query_coords):
        
        out = self.encoder(sensor_values)
        return self.decoder_1(out, query_coords)

    def training_step(self,batch, batch_idx):
        
        sensor_values, coords, field_values = batch
        
        # forward
        pred_values = self(sensor_values, coords)
        
        # loss
        loss = F.mse_loss(pred_values, field_values, reduction='sum')
        # loss = F.mse_loss(pred_values, field_values-torch.mean(field_values, dim=0), reduction='sum') # residual-style loss 
        
        self.log("train_loss", loss/field_values.numel(), 
                 on_step=True, on_epoch=True,prog_bar=True, logger=True,
                 batch_size=1)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        sensor_values, coords, field_values = batch
        # forward
        with torch.no_grad():
            pred_values = self(sensor_values, coords)
            field_values = torch.pow(10,field_values[:,0,0])
            pred_values = torch.pow(10,pred_values[:,0,0])
            # get metrics
            self.mae(pred_values, field_values)
            self.rmse(pred_values, field_values)
            self.r2_score(pred_values, field_values)

            self.log("val_r2", self.r2_score, on_epoch=True, logger=True)
            self.log("val_mae", self.mae, on_epoch=True, logger=True)
            self.log("val_rmse", self.rmse, on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)





    
    
    
    
    
    
    
    
    
