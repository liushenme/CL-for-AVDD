from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU, MaxPool3d, Module

from utils import Conv3d, Conv1d, ResConv3d
import torch.nn as nn
import torch

class ResC3DVideoEncoder(Module):
    """
    Video encoder (E_v): Process video frames to extract features.
    Input:
        V: (B, C, T, H, W)
    Output:
        F_v: (B, C_f, T)
    """

    def __init__(self, n_features=(64, 96, 128, 128)):
        super().__init__()

        n_dim0, n_dim1, n_dim2, n_dim3 = n_features

        # (B, 3, 512, 96, 96) -> (B, 64, 512, 32, 32)
        self.block0 = Sequential(
            Conv3d(3, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            ResConv3d(n_dim0, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 3, 3))
        )

        # (B, 64, 512, 32, 32) -> (B, 96, 512, 16, 16)
        self.block1 = Sequential(
            Conv3d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            ResConv3d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 96, 512, 16, 16) -> (B, 128, 512, 8, 8)
        self.block2 = Sequential(
            Conv3d(n_dim1, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            ResConv3d(n_dim2, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 128, 512, 8, 8) -> (B, 128, 512, 2, 2) -> (B, 512, 512) -> (B, 256, 512)
        self.block3 = Sequential(
            Conv3d(n_dim2, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            ResConv3d(n_dim3, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Rearrange("b c t h w -> b (c h w) t"),
            #Conv1d(n_dim3 * 4, 256, kernel_size=1, stride=1, build_activation=LeakyReLU)
            Conv1d(n_dim3 * 4, 128, kernel_size=1, stride=1, build_activation=LeakyReLU)
        )

    def forward(self, video: Tensor) -> Tensor:
        x = self.block0(video)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class C3DVideoEncoder(Module):
    """
    Video encoder (E_v): Process video frames to extract features.
    Input:
        V: (B, C, T, H, W)
    Output:
        F_v: (B, C_f, T)
    """

    def __init__(self, n_features=(64, 96, 128, 128)):
        super().__init__()

        n_dim0, n_dim1, n_dim2, n_dim3 = n_features

        # (B, 3, 512, 96, 96) -> (B, 64, 512, 32, 32)
        self.block0 = Sequential(
            Conv3d(3, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            #Conv3d(n_dim0, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 3, 3))
        )

        # (B, 64, 512, 32, 32) -> (B, 96, 512, 16, 16)
        self.block1 = Sequential(
            Conv3d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            #Conv3d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 96, 512, 16, 16) -> (B, 128, 512, 8, 8)
        self.block2 = Sequential(
            Conv3d(n_dim1, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            #Conv3d(n_dim2, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 128, 512, 8, 8) -> (B, 128, 512, 2, 2) -> (B, 512, 512) -> (B, 256, 512)
        self.block3 = Sequential(
            Conv3d(n_dim2, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Conv3d(n_dim3, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Rearrange("b c t h w -> b (c h w) t"),
            #Conv1d(n_dim3 * 4, 256, kernel_size=1, stride=1, build_activation=LeakyReLU)
            Conv1d(n_dim3 * 4, 128, kernel_size=1, stride=1, build_activation=LeakyReLU)
        )

    def forward(self, video: Tensor) -> Tensor:
        x = self.block0(video)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class LSTM_Encoder(nn.Module):
    def __init__(self,feature_dim,hidden_size,num_layers):
        super(LSTM_Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack_rnn = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_size, batch_first=False, bidirectional=False, num_layers=1)

    def forward(self, cur_inputs, current_frame):
        packed_input = nn.utils.rnn.pack_padded_sequence(cur_inputs, current_frame, enforce_sorted=False)
        rnn_out, _ = self.stack_rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, total_length=512) 
               
        return rnn_out

class c3d_lstm(nn.Module):
    def __init__(self):
        super(c3d_lstm, self).__init__()
        self.visualFrontend  = C3DVideoEncoder((64, 96, 128, 128))

        self.lstm = LSTM_Encoder(512, 512, 1)

        self.visualConv1D  = Conv1d(512, 128, kernel_size=1, stride=1, build_activation=LeakyReLU)

    def forward(self, x, frame):
        B, C, T, W, H = x.shape  
        #x = x.transpose(1, 2).view(B*T, 1, 1, W, H)
        #x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        #print('x', x.shape)
        #b, c ,t    
        x = x.transpose(1,2)     

        if type(frame) == torch.Tensor:
            frame = list(frame.detach().cpu().numpy())

        x = self.lstm(x.transpose(0, 1), frame).transpose(0, 1)
        
        #print(x.shape)        

        x = x.transpose(1,2)
        #x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)

        return x

class c3d_tcn(nn.Module):
    def __init__(self):
        super(c3d_lstm, self).__init__()
        self.visualFrontend  = C3DVideoEncoder((64, 96, 128, 128))

        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN

        self.visualConv1D  = visualConv1D()

    def forward(self, x, frame):
        B, C, T, W, H = x.shape  
        #x = x.transpose(1, 2).view(B*T, 1, 1, W, H)
        #x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        #print('x', x.shape)
        #b, c ,t    
        #print(x.shape)        
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)

        return x

class DSConv1d(nn.Module):
    def __init__(self):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(512),
            nn.Conv1d(512, 512, 1, bias=False),
            )

    def forward(self, x):
        out = self.net(x)
        return out + x

class visualConv1D(nn.Module):
    def __init__(self):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            )

    def forward(self, x):
        out = self.net(x)
        return out


class visualTCN(nn.Module):
    def __init__(self):
        super(visualTCN, self).__init__()
        stacks = []        
        for x in range(5):
            stacks += [DSConv1d()]
        self.net = nn.Sequential(*stacks) # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out

class visualConv1D(nn.Module):
    def __init__(self):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            )

    def forward(self, x):
        out = self.net(x)
        return out

