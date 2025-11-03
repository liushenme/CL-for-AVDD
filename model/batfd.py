from typing import Dict, Optional, Union, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import *
from .audio_encoder import CNNAudioEncoder
from .video_encoder import C3DVideoEncoder, c3d_lstm, ResC3DVideoEncoder

from torch import nn
import torch.nn.functional as F
#from .swin_transformer import build_swin_backbone


class Deepfakecla_bu_pred_cl(LightningModule):
    def __init__(self, exp = None,
        v_encoder: str = "c3d", a_encoder: str = "cnn", frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128), ae_features=(32, 64, 64), v_cla_feature_in=256, a_cla_feature_in=256,
        boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
        weight_frame_loss=2., weight_modal_bm_loss=1., weight_contrastive_loss=0.1, contrast_loss_margin=0.99,
        weight_decay=0.0001, learning_rate=0.0002, distributed=False
    ):
        super().__init__()
        #self.save_hyperparameters()

        
        self.video_encoder = C3DVideoEncoder(n_features=ve_features)
            
        self.audio_encoder = CNNAudioEncoder(n_features=ae_features)
            
        #self.fu_classifier = nn.Linear(256, 1)
        assert v_cla_feature_in == a_cla_feature_in
        #self.frame_loss = BCEWithLogitsLoss()
        self.frame_loss = CrossEntropyLoss()
        #self.mse = MSELoss()
        self.mse = MaskedFrameLoss(MSELoss())

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed


        self.pool = PoolAvg(256, 2)
        #self.pool = PoolAvg(256, 1) 

    def forward(self, video: Tensor, audio: Tensor, n_frames):
        # encoders
        #print("video", video.shape)
        #print("audio", audio.shape)

        v_features = self.video_encoder(video)
        #a_features = self.audio_encoder(audio)
        a_features = self.audio_encoder(audio)
        #fu_features = torch.cat((a_features_c, v_features_c), 2) 
        fu_features = torch.cat((a_features, v_features), 1) 
        #fu_features = self.selfAV(src = fu_features, tar = fu_features) 

        fu_features = fu_features.transpose(1, 2)
        #fu_cla = self.fu_classifier(fu_features)
        fu_cla = self.pool(fu_features, n_frames)
        
        #print(v_features.shape)
        sim = F.cosine_similarity(v_features, a_features, dim = 1)
        
        #print(sim.shape)
        #sim = sim[:, :n_frames]
        sim_mean = torch.mean(sim, dim= -1)

        #a_features = a_features[:, :, :n_frames]
        #v_features = v_features[:, :, :n_frames]

        return fu_cla, a_features, v_features, sim

    def forward_sim(self, video: Tensor, audio: Tensor, n_frames):
        # encoders
        #print("video", video.shape)
        #print("audio", audio.shape)

        v_features = self.video_encoder(video)
        #a_features = self.audio_encoder(audio)
        a_features = self.audio_encoder(audio)
        #print(a_features.shape)
        #print(v_features.shape)
            
        sim = F.cosine_similarity(v_features.squeeze(0), a_features.squeeze(0), dim = 0)
        sim = sim[:n_frames]
        sim_mean = torch.mean(sim)

        a_features = a_features[:, :, :n_frames]
        v_features = v_features[:, :, :n_frames]
        
        return a_features, v_features, sim, sim_mean 

    def loss_fn(self, fu_label: Tensor, fu_cla, v_features, a_features, n_frames) -> Dict[str, Tensor]:

        #fu_frame_loss = self.frame_loss(fu_cla.squeeze(1), fu_label)
        fu_frame_loss = self.frame_loss(fu_cla, fu_label)
        #contrast_loss = torch.clip(self.contrast_loss(v_features, a_features, fu_label, n_frames), max=1.)
        #loss = fu_frame_loss + 0.05 * contrast_loss
        loss = fu_frame_loss

        return loss
        #return {
        #    "loss": loss 
        #}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:
        batch_ori, batch_buffer = batch 
        video, audio, n_frames, fu_label, _ = batch_ori
        fu_cla, a_features, v_features, _ = self(video, audio, n_frames)
        #loss_ori = self.loss_fn(fu_label, fu_cla)
        loss_ori = self.loss_fn(fu_label, fu_cla, v_features, a_features, n_frames)

        if batch_buffer != None:
            video_buffer, audio_buffer, n_frames_buffer, fu_label_buffer, _, dense_buffer = batch_buffer
            a_features_buffer, v_features_buffer, sim_buffer = dense_buffer
            fu_cla_buffer, a_features, v_features, sim = self(video_buffer, audio_buffer, n_frames_buffer)
            #loss_buffer1 = self.loss_fn(fu_label_buffer, fu_cla_buffer)
            loss_buffer1 = self.loss_fn(fu_label_buffer, fu_cla_buffer, v_features_buffer, a_features_buffer, n_frames_buffer)
            #loss_buffer1 = self.loss_fn(fu_label_buffer, fu_cla_buffer)

            #print(a_features.shape)
            #print(a_features_buffer.shape)

            #loss_buffer_pred = self.mse(a_features, a_features_buffer, n_frames_buffer) + self.mse(v_features, v_features_buffer, n_frames_buffer)
            #loss_buffer_pred = self.mse(a_features, a_features_buffer, n_frames_buffer) + self.mse(v_features, v_features_buffer, n_frames_buffer) + self.mse2(sim, sim_buffer, n_frames_buffer)
            
            
            loss_buffer2 = self.mse(a_features, a_features_buffer, n_frames_buffer) + self.mse(v_features, v_features_buffer, n_frames_buffer)
            cat_sim = torch.cat((sim, sim_buffer))
            cat_y = fu_label_buffer.repeat(2)
            loss_con = sup_con_loss(cat_sim, 0.03, cat_y)
            #loss_con = sup_con_loss_no_norm(cat_sim, 0.03, cat_y)
            loss_buffer_pred = loss_buffer2 + loss_con
            
            loss_buffer = loss_buffer1 + 0.1 * loss_buffer_pred
            loss_total = loss_ori + loss_buffer

            #print('loss_ori', loss_ori)
            #print('loss_buffer1', loss_buffer1)
            #print('loss_buffer_pred', loss_buffer_pred)

            loss_dict = {
                        "loss": loss_total,
                        "loss_ori": loss_ori,
                        "loss_buffer": loss_buffer
                        }        
        else:
            loss_dict = {
                        "loss": loss_ori,
                        }        

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        batch_ori, batch_buffer = batch 
        video, audio, n_frames, fu_label, _ = batch_ori
        fu_cla, a_features, v_features, _ = self(video, audio, n_frames)
        loss_ori = self.loss_fn(fu_label, fu_cla, v_features, a_features, n_frames)

        loss_dict = {
                    "loss": loss_ori,
                    }        

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        video, audio, n_frames, fu_label = batch
        fu_cla = self(video, audio)
        return fu_cla

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }




