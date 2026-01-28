import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x_up = self.up_conv(x)
        x = torch.cat([x_up, skip], dim=1)
        return self.conv(x), x

class DecoderWithPrompt(nn.Module):
    def __init__(self, prompt_dim,out_channels=2):
        super(DecoderWithPrompt, self).__init__()
        self.up_trans = nn.ModuleList([
            UpBlock(512 + prompt_dim, 256),
            UpBlock(256 + prompt_dim + 128, 128),
            UpBlock(128 + prompt_dim + 64, 64),
        ])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.prompt_adapter = nn.Conv2d(prompt_dim, 256, kernel_size=1)

    def forward(self, feats, prompt):
        f512, f256, f128, f64 = feats
        b = f512.shape[0]


        prompt = prompt.view(prompt.shape[0], -1)  
        prompt = self.prompt_adapter(prompt.unsqueeze(-1).unsqueeze(-1)) 
        prompt_512 = F.interpolate(prompt, size=f512.shape[2:], mode='bilinear', align_corners=False)

        f512_prompt = torch.cat([f512, prompt_512], dim=1)
        out256, s256 = self.up_trans[0](f512_prompt, f256)

        prompt_256 = F.interpolate(prompt, size=out256.shape[2:], mode='bilinear', align_corners=False)
        f128_resized = F.interpolate(f128, size=out256.shape[2:], mode="bilinear", align_corners=False)
        out256_prompt = torch.cat([out256, prompt_256, f128_resized], dim=1)
        out128, s128 = self.up_trans[1](out256_prompt, f128)

        prompt_128 = F.interpolate(prompt, size=out128.shape[2:], mode='bilinear', align_corners=False)
        f64_resized = F.interpolate(f64, size=out128.shape[2:], mode="bilinear", align_corners=False)
        out128_prompt = torch.cat([out128, prompt_128, f64_resized], dim=1)
        out64, _ = self.up_trans[2](out128_prompt, f64)

        return self.final_conv(out64), out64

class ASF(nn.Module):
    def __init__(self, class_num, task_prompt, prompt_generator, use_anomaly_detection=False):
        super().__init__()
        self.disable_decoder_dropout = False

        self.class_num = class_num
        self.task_prompt = task_prompt
        self.prompt_generator = prompt_generator
        self.use_anomaly_detection = use_anomaly_detection
 
        self.prompt_linear_proj = nn.Identity()

        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool) 
        self.layer1 = encoder.layer1  
        self.layer2 = encoder.layer2 
        self.layer3 = encoder.layer3  
        self.layer4 = encoder.layer4  

        self._backbone_decoder = DecoderWithPrompt(prompt_dim=256, out_channels=class_num)
        self.organ_embedding = nn.Parameter(torch.randn(class_num, 256))

        self.train_metrics = {cls: [] for cls in range(class_num)}

    def segment_with_prompt(self, features, prompt):
        if prompt.dim() == 3:
            prompt = prompt.mean(dim=1) 

        prompt = self.prompt_linear_proj(prompt)
        output = self._backbone_decoder(features[:4], prompt)[0]
        return {'segmentation': output}

    def teacher_segment(self, features, prompt=None):
        with torch.no_grad():
            if prompt is None:
                prompt = self.task_prompt
                if isinstance(prompt, str):
                    prompt = torch.zeros(features[0].shape[0], 256, device=features[0].device)
                else:
                    prompt = prompt.expand(features[0].shape[0], -1)
            return self.segment_with_prompt(features[:4], prompt)['segmentation']

    def forward(self, x, setseq=None, use_anomaly_detection=None, return_aux=False):

        feats = self.forward_features(x)

        prompt_gen = getattr(self, "prompt_generator", None)
        if prompt_gen is not None:

            try:
                if setseq is None:
                    init_prompt, *_ = prompt_gen(feats)
                else:
                    init_prompt, *_ = prompt_gen(feats, setseq)
            except TypeError:

                init_prompt, *_ = prompt_gen(feats)

            B = x.size(0)
            C = getattr(self, "prompt_dim", 256)
            init_prompt = torch.zeros(B, C, device=x.device, dtype=feats[-1].dtype)

        out = self.segment_with_prompt(feats, init_prompt) 
        if return_aux:
            out["features"] = feats
            out["init_prompt"] = init_prompt
        return out

    def update_metrics(self, predictions, targets):
        with torch.no_grad():
            preds = torch.argmax(predictions, dim=1)
            for cls in range(self.class_num):
                pred_cls = (preds == cls).float()
                target_cls = (targets == cls).float()
                inter = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                dice = (2. * inter) / (union + 1e-8)
                self.train_metrics[cls].append(dice.item())



