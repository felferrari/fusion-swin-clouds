from torch import nn
import torch
from torchvision.models.swin_transformer import SwinTransformerBlock, PatchMerging
from .layers import ResidualBlock, PatchEmbed, PatchExpand, PatchExpandx4
from models.lib.SwinUnet.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from conf import general

class LateFusion(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(LateFusion, self).__init__()
        
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder_0 = ResUnetDecoder(depths)
        self.decoder_1 = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        #concatenate sources
        x_0 = self.encoder_0(x[0])
        x_1 = self.encoder_1(x[1])

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=1)
        x = self.classifier(x)
        return x

class ResUnetOpt(nn.Module):
    def __init__(self, input_depth, depths, n_classes):
        super(ResUnetOpt, self).__init__()
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x = torch.cat((x[0], x[1], x[4]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x

class ResUnetSAR(nn.Module):
    def __init__(self, input_depth, depths, n_classes):
        super(ResUnetSAR, self).__init__()
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x = torch.cat((x[2], x[3], x[4]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x

class EarlyFusion(nn.Module):
    def __init__(self, input_depth, depths, n_classes):
        super(EarlyFusion, self).__init__()
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x = torch.cat((x[0], x[1], x[2], x[3], x[4]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x

class LateFusion(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(LateFusion, self).__init__()
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder_0 = ResUnetDecoder(depths)
        self.decoder_1 = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(2*depths[0], n_classes)

    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0  = self.encoder_0(x_0)
        x_1  = self.encoder_1(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim = 1)

        x = self.classifier(x)

        return x

class JointFusionNoSkip(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(JointFusionNoSkip, self).__init__()
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder = ResUnetDecoderNoSkip(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = torch.cat((x_0[-1], x_1[-1]), dim=1)
        

        x = self.decoder(x)
        x = self.classifier(x)

        return x

class ResUnetEncoder(nn.Module):
    def __init__(self, input_depth, depths):
        super(ResUnetEncoder, self).__init__()
        self.first_res_block = nn.Sequential(
            nn.Conv2d(input_depth, depths[0], kernel_size=3, padding=1, padding_mode = 'reflect'),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], kernel_size=3, padding=1, padding_mode = 'reflect')
        )
        self.first_res_cov = nn.Conv2d(input_depth, depths[0], kernel_size=1)

        self.enc_block_0 = ResidualBlock(depths[0], depths[1], stride = 2)
        self.enc_block_1 = ResidualBlock(depths[1], depths[2], stride = 2)
        self.enc_block_2 = ResidualBlock(depths[2], depths[3], stride = 2)

    def forward(self, x):
        #first block
        x_idt = self.first_res_cov(x)
        x = self.first_res_block(x)
        x_0 = x + x_idt

        #encoder blocks
        x_1 = self.enc_block_0(x_0)
        x_2 = self.enc_block_1(x_1)
        x_3 = self.enc_block_2(x_2)

        return x_0, x_1, x_2, x_3
    
class ResUnetDecoder(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoder, self).__init__()
        self.dec_block_2 = ResidualBlock(depths[2] + depths[3], depths[2])
        self.dec_block_1 = ResidualBlock(depths[1] + depths[2], depths[1])
        self.dec_block_0 = ResidualBlock(depths[0] + depths[1], depths[0])

        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_0 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = x
        #concatenate sources
        x_2u = self.upsample_2(x_3)
        x_2c = torch.cat((x_2u, x_2), dim=1)
        x_2 = self.dec_block_2(x_2c)

        x_1u = self.upsample_1(x_2)
        x_1c = torch.cat((x_1u, x_1), dim=1)
        x_1 = self.dec_block_1(x_1c)

        x_0u = self.upsample_0(x_1)
        x_0c = torch.cat((x_0u, x_0), dim=1)
        x_0 = self.dec_block_0(x_0c)

        return x_0

class ResUnetDecoderNoSkip(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoderNoSkip, self).__init__()
        self.dec_block_2 = ResidualBlock(2*depths[3], depths[2])
        self.dec_block_1 = ResidualBlock(depths[2], depths[1])
        self.dec_block_0 = ResidualBlock(depths[1], depths[0])

        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_0 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_3 = x
        x_2u = self.upsample_2(x_3)
        x_2 = self.dec_block_2(x_2u)

        x_1u = self.upsample_1(x_2)
        x_1 = self.dec_block_1(x_1u)

        x_0u = self.upsample_0(x_1)
        x_0 = self.dec_block_0(x_0u)

        return x_0

class ResUnetClassifier(nn.Module):
    def __init__(self, depth, n_classes):
        super(ResUnetClassifier, self).__init__()
        self.res_block = ResidualBlock(depth, depth)
        self.last_conv = nn.Conv2d(depth, n_classes, kernel_size=1)
        self.last_act = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.res_block(x)
        x = self.last_conv(x)
        x = self.last_act(x)
        return x

class SwinUnetOptOld(nn.Module):
    def __init__(self, input_depth, n_classes) -> None:
        super(SwinUnetOptOld, self).__init__()
        self.model = SwinTransformerSys(
            img_size=general.PATCH_SIZE,
            in_chans = input_depth,
            num_classes = n_classes,
            window_size = 4
        )
    
    def forward(self, x):
        x = torch.cat((x[0], x[1], x[4]), dim=1)
        return self.model(x)

class SwinUnetOpt(nn.Module):
    def __init__(self, input_depth, n_classes) -> None:
        super(SwinUnetOpt, self).__init__()
        self.encoder = SwinEncoder(
            input_depth=input_depth,
            base_dim=general.SWIN_BASE_DIM,
            n_heads=general.SWIN_N_HEADS
            )
    
        self.decoder = SwinDecoder(
            base_dim=general.SWIN_BASE_DIM,
            n_heads=general.SWIN_N_HEADS
            )
        self.classifier = ResUnetClassifier(general.SWIN_BASE_DIM, n_classes)
    
    def forward(self, x):
        x = torch.cat((x[0], x[1], x[4]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x
    
class SwinUnetSAR(nn.Module):
    def __init__(self, input_depth, n_classes) -> None:
        super(SwinUnetSAR, self).__init__()
        self.encoder = SwinEncoder(
            input_depth=input_depth,
            base_dim=general.SWIN_BASE_DIM,
            n_heads=general.SWIN_N_HEADS
            )
    
        self.decoder = SwinDecoder(
            base_dim=general.SWIN_BASE_DIM,
            n_heads=general.SWIN_N_HEADS
            )
        self.classifier = ResUnetClassifier(general.SWIN_BASE_DIM, n_classes)
    
    def forward(self, x):
        x = torch.cat((x[2], x[3], x[4]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x

class SwinEncoder(nn.Module):
    def __init__(self, input_depth, base_dim, n_heads) -> None:
        super(SwinEncoder, self).__init__()
        self.embed = PatchEmbed(
            img_size=[general.PATCH_SIZE,general.PATCH_SIZE],
            patch_size=general.SWIN_PATCH_SIZE,
            in_chans=input_depth,
            embed_dim=base_dim
            )
        self.msa_0_0 = SwinTransformerBlock(
            dim = base_dim, 
            num_heads = n_heads[0], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_0_1 = SwinTransformerBlock(
            dim = base_dim, 
            num_heads = n_heads[0], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=general.SWIN_SHIFT_SIZE 
            )
        self.merge_0 = PatchMerging(base_dim)

        self.msa_1_0 = SwinTransformerBlock(
            dim = 2*base_dim, 
            num_heads = n_heads[1], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_1_1 = SwinTransformerBlock(
            dim = 2*base_dim, 
            num_heads = n_heads[1], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=general.SWIN_SHIFT_SIZE 
            )
        self.merge_1 = PatchMerging(2*base_dim)

        self.msa_2_0 = SwinTransformerBlock(
            dim = 4*base_dim, 
            num_heads = n_heads[2], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_2_1 = SwinTransformerBlock(
            dim = 4*base_dim, 
            num_heads = n_heads[2], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=general.SWIN_SHIFT_SIZE 
            )
        self.merge_2 = PatchMerging(4*base_dim)

        self.msa_3_0 = SwinTransformerBlock(
            dim = 8*base_dim, 
            num_heads = n_heads[3], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_3_1 = SwinTransformerBlock(
            dim = 8*base_dim, 
            num_heads = n_heads[3], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )

    def forward(self, x):
        x = self.embed(x)
        x = x.permute((0,2,3,1))
        x = self.msa_0_0(x)
        x_0 = self.msa_0_1(x)
        x = self.merge_0(x_0)

        x = self.msa_1_0(x)
        x_1 = self.msa_1_1(x)
        x = self.merge_1(x_1)

        x = self.msa_2_0(x)
        x_2 = self.msa_2_1(x)
        x = self.merge_2(x_2)

        x = self.msa_3_0(x)
        x_3 = self.msa_3_1(x)
        return x_3, x_2, x_1, x_0

class SwinDecoder(nn.Module):
    def __init__(self, base_dim, n_heads) -> None:
        super(SwinDecoder, self).__init__()
        self.patch_expand_2 = PatchExpand(dim = 8*base_dim)
        self.patch_expand_1 = PatchExpand(dim = 4*base_dim)
        self.patch_expand_0 = PatchExpand(dim = 2*base_dim)

        self.patch_expand_last = PatchExpandx4(dim = base_dim)


        self.msa_0_0 = SwinTransformerBlock(
            dim = base_dim, 
            num_heads = n_heads[0], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_0_1 = SwinTransformerBlock(
            dim = base_dim, 
            num_heads = n_heads[0], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=general.SWIN_SHIFT_SIZE 
            )

        self.msa_1_0 = SwinTransformerBlock(
            dim = 2*base_dim, 
            num_heads = n_heads[1], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_1_1 = SwinTransformerBlock(
            dim = 2*base_dim, 
            num_heads = n_heads[1], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=general.SWIN_SHIFT_SIZE 
            )

        self.msa_2_0 = SwinTransformerBlock(
            dim = 4*base_dim, 
            num_heads = n_heads[2], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=[0,0] 
            )
        self.msa_2_1 = SwinTransformerBlock(
            dim = 4*base_dim, 
            num_heads = n_heads[2], 
            window_size= general.SWIN_WINDOW_SIZE,
            shift_size=general.SWIN_SHIFT_SIZE 
            )
        
        self.skip_0 = nn.Conv2d( 2*base_dim,  base_dim, kernel_size=1)
        self.skip_1 = nn.Conv2d( 4*base_dim,  2*base_dim, kernel_size=1)
        self.skip_2 = nn.Conv2d( 8*base_dim,  4*base_dim, kernel_size=1)
        

    def forward(self, x):
        x_3, x_2s, x_1s, x_0s = x
        x_2e = self.patch_expand_2(x_3)
        x_2 = torch.cat((x_2e, x_2s), dim=-1)
        x_2 = self.skip_2(x_2.permute((0,3,1,2))).permute((0,2,3,1))
        x_2 = self.msa_2_0(x_2)
        x_2 = self.msa_2_1(x_2)
        x_1e = self.patch_expand_1(x_2)
        x_1 = torch.cat((x_1e, x_1s), dim=-1)
        x_1 = self.skip_1(x_1.permute((0,3,1,2))).permute((0,2,3,1))
        x_1 = self.msa_1_0(x_1)
        x_1 = self.msa_1_1(x_1)
        x_0e = self.patch_expand_0(x_1)
        x_0 = torch.cat((x_0e, x_0s), dim=-1)
        x_0 = self.skip_0(x_0.permute((0,3,1,2))).permute((0,2,3,1))
        x_0 = self.msa_0_0(x_0)
        x_0 = self.msa_0_1(x_0)
        x_0 = self.patch_expand_last(x_0)
        
        return x_0

class SwinUnetSAROld(nn.Module):
    def __init__(self, input_depth, n_classes) -> None:
        super(SwinUnetSAROld, self).__init__()
        self.model = SwinTransformerSys(
            img_size=general.PATCH_SIZE,
            in_chans = input_depth,
            num_classes = n_classes,
            window_size = 4
        )
    
    def forward(self, x):
        x = torch.cat((x[2], x[3], x[4]), dim=1)
        return self.model(x)

class SwinUnetEF(nn.Module):
    def __init__(self, input_depth, n_classes) -> None:
        super(SwinUnetEF, self).__init__()
        self.model = SwinTransformerSys(
            img_size=general.PATCH_SIZE,
            in_chans = input_depth,
            num_classes = n_classes,
            window_size = 4
        )
    
    def forward(self, x):
        x = torch.cat((x[0], x[1], x[2], x[3], x[4]), dim=1)
        return self.model(x)