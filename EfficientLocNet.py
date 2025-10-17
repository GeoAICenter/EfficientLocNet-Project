from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from LocNet_Data_Loader import RadioMapDataset
from tqdm import tqdm


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.spatial_wise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.channel_wise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.spatial_wise.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.channel_wise.weight, mode='fan_out')

    def forward(self, x):
        x = self.spatial_wise(x)
        x = self.channel_wise(x)
        return x


class PyramidExtraction(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(input_size, input_size, kernel_size=1)
        self.conv_atrocious_r1 = DepthWiseSeparableConv(input_size, input_size, kernel_size=(7, 7), padding=3, bias=True)
        self.conv_atrocious_r2 = DepthWiseSeparableConv(input_size, input_size, kernel_size=(7, 7), padding=9, dilation=3, bias=True)
        self.conv_atrocious_r4 = DepthWiseSeparableConv(input_size, input_size, kernel_size=(7, 7), padding=27, dilation=9, bias=True)
        self.conv_out = nn.Conv2d(4*(input_size), input_size, kernel_size=1)

    def forward(self, in_features):
        x_1x1 = self.conv_1x1(in_features)
        x_atrocious_r1 = self.conv_atrocious_r1(in_features)
        x_atrocious_r2 = self.conv_atrocious_r2(in_features)
        x_atrocious_r4 = self.conv_atrocious_r4(in_features)
        x = torch.cat([x_1x1, x_atrocious_r1, x_atrocious_r2, x_atrocious_r4], dim=1)
        mask = nn.Sigmoid()(self.conv_out(x))
        return in_features + in_features * mask


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=3):
        super().__init__()
        self.conv = DepthWiseSeparableConv(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)
        return x


def convSepNormLeak(in_channels, out_channels, alpha):
    return nn.Sequential(
        ConvNorm(in_channels, out_channels, kernel_size=(7, 7), padding=3),
        nn.LeakyReLU(alpha)
    )


class Encoder(nn.Module):
    def __init__(self, input_size, enc_out, dim, alpha):
        super().__init__()
        self.alpha = alpha
        self.l_in = convSepNormLeak(input_size, dim, alpha)
        self.l1 = convSepNormLeak(dim, dim, alpha)
        self.l2 = convSepNormLeak(dim, dim, alpha)
        self.pyramid_1 = PyramidExtraction(dim)

        self.l3 = convSepNormLeak(dim, dim, alpha)
        self.l4 = convSepNormLeak(dim, dim, alpha)
        self.pyramid_2 = PyramidExtraction(dim)

        self.l5 = convSepNormLeak(dim, dim, alpha)
        self.l6 = convSepNormLeak(dim, dim, alpha)
        self.pyramid_3 = PyramidExtraction(dim)

        self.l7 = convSepNormLeak(dim, enc_out, alpha)

        self.downsampled1 = nn.AvgPool2d(2, 2)
        self.downsampled2 = nn.AvgPool2d(2, 2)
        self.downsampled3 = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.l_in(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.pyramid_1(x)
        skip_1 = x

        x = self.downsampled1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.pyramid_2(x)
        skip_2 = x

        x = self.downsampled2(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.pyramid_3(x)
        skip_3 = x

        x = self.downsampled3(x)
        x = self.l7(x)
        return x, skip_1, skip_2, skip_3


class Decoder(nn.Module):
    def __init__(self, input_size, output, dim, alpha):
        super().__init__()
        self.l_in = convSepNormLeak(input_size, input_size, alpha)
        self.l1 = convSepNormLeak(input_size + dim, dim, alpha)
        self.l2 = convSepNormLeak(dim, dim, alpha)
        self.l3 = convSepNormLeak(dim, dim, alpha)
        self.l4 = convSepNormLeak(dim + dim, dim, alpha)
        self.l5 = convSepNormLeak(dim, dim, alpha)
        self.l6 = convSepNormLeak(dim, dim, alpha)
        self.l7 = convSepNormLeak(dim + dim, dim, alpha)
        self.l8 = convSepNormLeak(dim, dim, alpha)
        self.l9 = convSepNormLeak(dim, dim, alpha)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.out_conv = DepthWiseSeparableConv(dim, output, kernel_size=(3, 3), padding=1)

    def forward(self, x, skip_1, skip_2, skip_3):
        x = self.l_in(x)
        x = self.upsample1(x)

        x = torch.cat([x, skip_3], dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.upsample2(x)

        x = torch.cat([x, skip_2], dim=1)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.upsample3(x)

        x = torch.cat([x, skip_1], dim=1)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.out_conv(x)
        return x


class SelfAttn(nn.Module):
    def __init__(self, input_size, output_size, dim, heads):
        super().__init__()
        self.Q = DepthWiseSeparableConv(input_size, dim * heads, kernel_size=(7, 7), padding=3, bias=True)
        self.K = DepthWiseSeparableConv(input_size, dim * heads, kernel_size=(7, 7), padding=3, bias=True)
        self.V = DepthWiseSeparableConv(input_size, dim * heads, kernel_size=(7, 7), padding=3, bias=True)
        self.conv = nn.Conv2d(dim * heads, output_size, kernel_size=1)
        self.heads = heads
        self.dim = dim
        self._init_weight()

    def _init_weight(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x):
        B, C, H, W = x.shape
        proj_q = torch.transpose(self.Q(x).view(-1, self.dim, H*W).contiguous(), 1, 2).contiguous()
        proj_k = self.K(x).view(-1, self.dim, H*W).contiguous()
        proj_v = self.V(x).view(-1, self.dim, H*W).contiguous()
        attention_map = nn.Softmax(dim=-1)(torch.bmm(proj_q, proj_k))
        output = torch.bmm(
            proj_v,
            torch.transpose(attention_map, 1, 2).contiguous()
        ).view(-1, self.heads * self.dim, H, W).contiguous()
        output = self.conv(output)
        return output


class SEPASA(nn.Module):
    def __init__(self, enc_in, enc_out, dim, attention_dims, dec_in, output_dim, heads, alpha):
        super().__init__()
        self.Encoder = Encoder(enc_in, enc_out, dim, alpha)
        self.Attn = SelfAttn(enc_out, dec_in, attention_dims, heads)
        self.Decoder = Decoder(dec_in, output_dim, dim, alpha)

    def forward(self, x):
        x, skip_1, skip_2, skip_3 = self.Encoder(x)
        x = self.Attn(x)
        return self.Decoder(x, skip_1, skip_2, skip_3)


def FocalLoss(pred, y, gamma, alpha):
    BCE = nn.functional.binary_cross_entropy_with_logits(pred, y, reduction='none')
    pred_y = torch.exp(-BCE) * y + (1.0 - torch.exp(-BCE)) * (1.0 - y)
    z = (1.0 - pred_y) * y
    z += (1.0 - y) * pred_y
    z = z**gamma
    scale = y * alpha
    scale += (1.0 - y) * (1.0 - alpha)
    return torch.sum(z * BCE * scale)


def evaluation(preds, targets):
    preds = preds.view(-1, preds.shape[2] * preds.shape[3]).contiguous()
    targets = targets.view(-1, targets.shape[2] * targets.shape[3]).contiguous()
    preds_loc = torch.argmax(preds, dim=1)
    targets_loc = torch.argmax(targets, dim=1)
    return torch.sqrt((preds_loc // 256 - targets_loc // 256)**2 + (preds_loc % 256 - targets_loc % 256)**2).sum()


class SEPASALit(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SEPASA(1, 32, 32, 32, 32, 1, 1, 0.3)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        predictions = self.model(data)
        loss = FocalLoss(predictions, target, 3.0, 0.75)
        self.log('Focal_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        predictions = self.model(data)
        dist_loss = evaluation(predictions, target)
        self.log('val_loss', dist_loss, sync_dist=True, reduce_fx=torch.sum)
        return dist_loss

    def on_validation_epoch_end(self):
        sum_val = self.trainer.callback_metrics.get('val_loss')
        self.log('dist_loss', sum_val, sync_dist=True)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    checkpoint_callback = ModelCheckpoint(
        monitor="dist_loss",
        mode="min",
        save_top_k=-1,
        filename="AAAAA_NO_BUILDING_MSSA_CHOSEN-{epoch:02d}-{dist_loss:.2f}",
        dirpath="/home/xxx/SOTA_Third_project"
    )
    BATCH = 16
    Train = RadioMapDataset('/home/xxx', 'Train_0_0001_TO_0_001')
    Val = RadioMapDataset('/home/xxx', 'Val_0_0001_TO_0_001')
    Train_Loader = DataLoader(
        Train,
        shuffle=True,
        batch_size=BATCH,
        drop_last=True,
        num_workers=4 * 4
    )
    Val_Loader = DataLoader(
        Val,
        batch_size=BATCH,
        num_workers=4 * 4,
    )
    L_model = SEPASALit()
    trainer = L.Trainer(accelerator="gpu", max_epochs=200, devices=[3,4,5,6], callbacks=[checkpoint_callback])
    trainer.fit(L_model, Train_Loader, Val_Loader)

#    model = SEPASA(2, 32, 32, 32, 32, 1, 1, 0.3)
#    from torch.utils.flop_counter import FlopCounterMode
#    def get_flops(model, inp, with_backward=False):
#        model.eval()
#        inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)
    
#        flop_counter = FlopCounterMode(display=False)
#        with flop_counter:
#            if with_backward:
#                model(inp).sum().backward()
#            else:
#                x = model(inp)
#        print(x.shape)
#        total_flops = flop_counter.get_total_flops()
#        return total_flops
#    print(get_flops(model, (1, 2, 256, 256)))
    
#    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
#    print(params)


