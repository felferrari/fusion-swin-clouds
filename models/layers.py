from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride = 1):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride = stride, padding=1, padding_mode = 'zeros'),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, padding_mode = 'zeros')
        )
        self.idt_conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride = stride, padding_mode = 'zeros')

    def forward(self, x):
        x_idt = self.idt_conv(x)
        x = self.res_block(x)

        return x + x_idt
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #self.proj = Conv2D(embed_dim, kernel_size=patch_size,
        #                   strides=patch_size, name='proj')
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias = False)
        

    def forward(self, x):
        x = self.proj(x)
        
        return x
    
class PatchExpand(nn.Module):
    def __init__(self, dim: int, norm_layer = nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        if norm_layer is not None:
            self.norm_layer = norm_layer(2*dim)
        else:
            self.norm_layer = None
        

    def forward(self, x):
        x = self.expand(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = x.permute((0,3,1,2))
        x = nn.functional.pixel_shuffle(x, 2)
        x = x.permute((0,2,3,1))
        
        return x

class PatchExpandx4(nn.Module):
    def __init__(self, dim: int, norm_layer = nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        if norm_layer is not None:
            self.norm_layer = norm_layer(16 * dim)
        else:
            self.norm_layer = None
        

    def forward(self, x):
        x = self.expand(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = x.permute((0,3,1,2))
        x = nn.functional.pixel_shuffle(x, 4)
        x = x.permute((0,2,3,1))
        
        return x