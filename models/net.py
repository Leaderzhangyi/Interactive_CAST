import torch.nn as nn
import torch

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)), # [1, 3, 256, 256]
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), # [1, 64, 128, 128] 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), #  [1, 128, 64, 64] 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4  #  [1, 256, 64, 64] 
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), #  [1, 256, 32, 32] 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used  [1, 512, 32, 32] 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class ADAIN_Encoder(nn.Module):
    def __init__(self, encoder, gpu_ids=[]):
        super(ADAIN_Encoder, self).__init__()
        enc_layers = list(encoder.children()) # self.netAE = net.ADAIN_Encoder(vgg, self.gpu_ids)
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1 64
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1 128
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1 256
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1 512 [batch_size,3,32,32]
        # 以上是3 * 256 * 256 -> 512 * 32 * 32
        self.mse_loss = nn.MSELoss()

        # 经过笔触模块 
        self.texture = nn.Sequential(
            nn.Conv2d(in_channels = 512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(512, 256, (3, 3)),
            # nn.ReLU(), #  [1, 256, 32, 32] 
            # nn.Upsample(scale_factor=2, mode='nearest') # [1, 256, 64, 64] 
        ) 
        self.test = nn.Sequential(
            nn.Conv2d(in_channels = 512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU()
        )

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4','texture','test']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content, style, encoded_only = False,texture = False):
        # print("输入内容图大小:",content,content.size())
        # print("输入内容图大小:",style,style.size())

        # 通过enc_4 得到 512 * 32 * 32
        style_feats = self.encode_with_intermediate(style) 
        content_feats = self.encode_with_intermediate(content)

        # 通过texture unit 得到  256 * 64 * 64
        # # 一旦修改了层数 loss就有问题 ？ 原因？
        
        # style_feats = self.test(style_feats[-1])
        # content_feats = self.test(content_feats[-1])


        if encoded_only:
            return content_feats[-1], style_feats[-1]
        # elif texture:
        #     adain_feat = self.adain(content_feats[-1], style_feats[-1])
        #     res = self.texture(adain_feat)
        #     return res 
        else:
            adain_feat = self.adain(content_feats[-1], style_feats[-1]) # origin
            #adain_feat = self.adain(content_feats, style_feats)
            #print("adain_size = ",adain_feat.size())
            
            return self.test(adain_feat)
            #return adain_feat


class Decoder(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Decoder, self).__init__()
        decoder = [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(), # 256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),# 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),# 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
            ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, adain_feat):
        fake_image = self.decoder(adain_feat)

        return fake_image

# 思路

# 1. 准备256，512，800三个文件夹数据集
#   每个文件夹都对应 train_A 和 train_B
# 2. 对三个文件夹进行分别训练，用state_dict()存储权重
# 3. test时怎么用参数控制？
#