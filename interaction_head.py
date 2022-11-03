"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict

import pocket

from ops import compute_spatial_encodings

class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    hidden_state_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        hidden_state_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(hidden_state_size / cardinality)
        assert sub_repr_size * cardinality == hidden_state_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, hidden_state_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))

class ModifiedEncoderLayer(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        if representation_size % num_heads != 0:
            raise ValueError(
                f"The given representation size {representation_size} "
                f"should be divisible by the number of attention heads {num_heads}."
            )
        self.sub_repr_size = int(representation_size / num_heads)

        self.hidden_size = hidden_size
        self.representation_size = representation_size

        self.num_heads = num_heads
        self.return_weights = return_weights

        self.unary = nn.Linear(hidden_size, representation_size)
        self.pairwise = nn.Linear(representation_size, representation_size)
        self.attn = nn.ModuleList([nn.Linear(3 * self.sub_repr_size, 1) for _ in range(num_heads)])
        self.message = nn.ModuleList([nn.Linear(self.sub_repr_size, self.sub_repr_size) for _ in range(num_heads)])
        self.aggregate = nn.Linear(representation_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.ffn = pocket.models.FeedForwardNetwork(hidden_size, hidden_size * 4, dropout_prob)

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_repr_size
        )
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError("Incorrect tensor shape")

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        device = x.device
        n = len(x)

        u = F.relu(self.unary(x))
        p = F.relu(self.pairwise(y))

        # Unary features (H, N, L)
        u_r = self.reshape(u)
        # Pairwise features (H, N, N, L)
        p_r = self.reshape(p)

        i, j = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )

        # Features used to compute attention (H, N, N, 3L)
        attn_features = torch.cat([
            u_r[:, i], u_r[:, j], p_r
        ], dim=-1)
        # Attention weights (H,) (N, N, 1)
        weights = [
            F.softmax(l(f), dim=0) for f, l
            in zip(attn_features, self.attn)
        ]
        # Repeated unary feaures along the third dimension (H, N, N, L)
        u_r_repeat = u_r.unsqueeze(dim=2).repeat(1, 1, n, 1)
        messages = [
            l(f_1 * f_2) for f_1, f_2, l
            in zip(u_r_repeat, p_r, self.message)
        ]

        aggregated_messages = self.aggregate(F.relu(
            torch.cat([
                (w * m).sum(dim=0) for w, m
                in zip(weights, messages)
            ], dim=-1)
        ))
        aggregated_messages = self.dropout(aggregated_messages)
        x = self.norm(x + aggregated_messages)
        x = self.ffn(x)

        if self.return_weights: attn = weights
        else: attn = None

        return x, attn

class ModifiedEncoder(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, num_layers: int = 2,
        dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([ModifiedEncoderLayer(
            hidden_size=hidden_size, representation_size=representation_size,
            num_heads=num_heads, dropout_prob=dropout_prob, return_weights=return_weights
        ) for _ in range(num_layers)])
    
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        for layer in self.mod_enc:
            x, attn = layer(x, y)
            attn_weights.append(attn)
        return x, attn_weights

class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_pair_predictor: nn.Module
        Module that classifies box pairs
    hidden_state_size: int
        Size of the object features
    representation_size: int
        Size of the human-object pair features
    num_channels: int
        Number of channels in the global image features
    num_classes: int
        Number of target classes
    human_idx: int
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
    """
    def __init__(self,
        box_pair_predictor: nn.Module,
        hidden_state_size: int, representation_size: int,
        num_channels: int, num_classes: int, human_idx: int,
        object_class_to_target_class: List[list]
    ) -> None:
        super().__init__()

        self.box_pair_predictor = box_pair_predictor

        self.hidden_state_size = hidden_state_size
        self.representation_size = representation_size

        self.num_classes = num_classes
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        # Map spatial encodings to the same dimension as appearance features
        # self.spatial_head = nn.Sequential(
        #     nn.Linear(36, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, representation_size),
        #     nn.ReLU(),
        # )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, representation_size),
            nn.ReLU(),
        )

        self.coop_layer = ModifiedEncoder(
            hidden_size=hidden_state_size,
            representation_size=representation_size,
            num_layers=2,
            return_weights=True
        )
        self.comp_layer = pocket.models.TransformerEncoderLayer(
            hidden_size=representation_size * 2,
            return_weights=True
        )

        self.mbf = MultiBranchFusion(
            hidden_state_size * 2,
            representation_size, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mbf_g = MultiBranchFusion(
            num_channels, representation_size,
            representation_size, cardinality=16
        )

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        # # ver1
        # self.dec1_1 = CBR2d(in_channels=2048, out_channels=1024)
        # self.dec1_2 = CBR2d(in_channels=1024, out_channels=512)
        #
        # self.unpool1 = nn.ConvTranspose2d(in_channels=512, out_channels= 512, kernel_size=2, stride=2, padding=0, bias=True)
        # self.dec2_1 = CBR2d(in_channels=512, out_channels=256)
        # self.dec2_2 = CBR2d(in_channels=256, out_channels=128)
        #
        # self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        # self.dec3_1 = CBR2d(in_channels=128, out_channels=64)
        # self.dec3_2 = CBR2d(in_channels=64, out_channels=32)
        #
        # self.unpool3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=True)
        # self.dec4_1 = CBR2d(in_channels=32, out_channels=16)
        #
        # self.enc4_1 = CBR2d(in_channels=16, out_channels=32)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc3_1 = CBR2d(in_channels=2 * 32, out_channels= 64)
        # self.enc3_2 = CBR2d(in_channels=64, out_channels=128)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc2_1 = CBR2d(in_channels=2* 128, out_channels= 256)
        # self.enc2_2 = CBR2d(in_channels=256, out_channels=512)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc1_1 = CBR2d(in_channels=2 * 512, out_channels= 1024)
        # self.enc1_2 = CBR2d(in_channels=1024, out_channels=2048)
        # # ver 2
        # self.dec1_1 = CBR2d(in_channels=2048, out_channels=2048)
        # self.dec1_2 = CBR2d(in_channels=2048, out_channels=1024)
        #
        # self.unpool1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0,
        #                                   bias=True)
        # self.dec2_1 = CBR2d(in_channels=1024, out_channels=1024)
        # self.dec2_2 = CBR2d(in_channels=1024, out_channels=512)
        #
        # self.unpool2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0,
        #                                   bias=True)
        # self.dec3_1 = CBR2d(in_channels=512, out_channels=512)
        # self.dec3_2 = CBR2d(in_channels=512, out_channels=256)
        #
        # self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
        #                                   bias=True)
        # self.dec4_1 = CBR2d(in_channels=256, out_channels=256)
        #
        # self.enc4_1 = CBR2d(in_channels=256, out_channels=256)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc3_1 = CBR2d(in_channels=2 * 256, out_channels=512)
        # self.enc3_2 = CBR2d(in_channels=512, out_channels=512)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc2_1 = CBR2d(in_channels=2 * 512, out_channels=1024)
        # self.enc2_2 = CBR2d(in_channels=1024, out_channels=1024)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc1_1 = CBR2d(in_channels=2 * 1024, out_channels=2048)
        # self.enc1_2 = CBR2d(in_channels=2048, out_channels=2048)

        # ver3
        self.dec1_1 = CBR2d(in_channels=2048, out_channels=2048)
        self.dec1_2 = CBR2d(in_channels=2048, out_channels=1024)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0,
                                          bias=True)

        self.dec2_1 = CBR2d(in_channels=1024, out_channels=512)
        self.enc2_1 = CBR2d(in_channels=512, out_channels=1024)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc1_1 = CBR2d(in_channels= 2 * 1024, out_channels=2048)
        self.enc1_2 = CBR2d(in_channels=2048, out_channels=2048)




        # # #test
        # self.dec1 = CBR2d(in_channels=2048, out_channels=1024)
        # self.dec1_unpool = nn.ConvTranspose2d(in_channels=1024, out_channels=1024,kernel_size=2,stride=2,padding=0,bias=True)
        #
        # self.dec2 = CBR2d(in_channels=1024, out_channels=1024)
        # self.enc2 = CBR2d(in_channels=1024, out_channels=1024)
        # self.enc2_pool = nn.MaxPool2d(kernel_size=2)
        #
        # self.enc1 = CBR2d(in_channels= 2* 1024, out_channels= 2048)


    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def forward(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: Tensor
            (B, 2) Image shapes, heights followed by widths
        region_props: List[dict]
            Region proposals with the following keys
            `boxes`: Tensor
                (N, 4) Bounding boxes
            `scores`: Tensor
                (N,) Object confidence scores
            `labels`: Tensor
                (N,) Object class indices
            `hidden_states`: Tensor
                (N, 256) Object features
        """

        device = features.device
        # # wish to run(ver2)
        # dec1_1 = self.dec1_1(features)
        # dec1_2 = self.dec1_2(dec1_1)
        # unpool1 = self.unpool1(dec1_2)
        #
        # dec2_1 = self.dec2_1(unpool1)
        # dec2_2 = self.dec2_2(dec2_1)
        # unpool2 = self.unpool2(dec2_2)
        #
        # dec3_1 = self.dec3_1(unpool2)
        # dec3_2 = self.dec3_2(dec3_1)
        # unpool3 = self.unpool3(dec3_2)
        #
        # dec4_1 = self.dec4_1(unpool3)
        # enc4_1 = self.enc4_1(dec4_1)
        #
        # pool3 = self.pool3(enc4_1)
        # cat3 = torch.cat((dec3_2, pool3), dim=1)
        # enc3_1 = self.enc3_1(cat3)
        # enc3_2 = self.enc3_2(enc3_1)
        #
        # pool2 = self.pool2(enc3_2)
        # cat2 = torch.cat((dec2_2, pool2), dim=1)
        # enc2_1 = self.enc2_1(cat2)
        # enc2_2 = self.enc2_2(enc2_1)
        #
        # pool1 = self.pool1(enc2_2)
        # cat1 = torch.cat((dec1_2, pool1), dim=1)
        # enc1_1 = self.enc1_1(cat1)
        # enc1_2 = self.enc1_2(enc1_1)

        # ver3
        dec1_1 = self.dec1_1(features)
        dec1_2 = self.dec1_2(dec1_1)
        unpool1 = self.unpool1(dec1_2)

        dec2_1 = self.dec2_1(unpool1)
        enc2_1 = self.enc2_1(dec2_1)

        pool1 = self.pool1(enc2_1)
        cat1 = torch.cat((dec1_2, pool1), dim=1)

        enc1_1 = self.enc1_1(cat1)
        enc1_2 = self.enc1_2(enc1_1)

        # features = enc1_2
        global_features = self.avg_pool(enc1_2).flatten(start_dim=1)

        # # test
        # dec1 = self.dec1(features)
        # unpool = self.dec1_unpool(dec1)
        #
        # dec2 = self.dec2(unpool)
        # enc2 = self.enc2(dec2)
        #
        # pool = self.enc2_pool(enc2)
        # cat = torch.cat((pool,dec1), dim=1)
        #
        # enc1 = self.enc1(cat)
        # #features = enc1
        # global_features = self.avg_pool(enc1).flatten(start_dim=1)



        #global_features = self.avg_pool(features).flatten(start_dim=1)

        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        pairwise_tokens_collated = []
        attn_maps_collated = []

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; unary_tokens = unary_tokens[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                pairwise_tokens_collated.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()

            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [boxes[x]], [boxes[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            # Reshape the spatial features
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n, n, -1)

            # Run the cooperative layer
            unary_tokens, unary_attn = self.coop_layer(unary_tokens, box_pair_spatial_reshaped)
            # Generate pairwise tokens with MBF
            pairwise_tokens = torch.cat([
                self.mbf(
                    torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep]
                ), self.mbf_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep])
            ], dim=1)
            # Run the competitive layer
            pairwise_tokens, pairwise_attn = self.comp_layer(pairwise_tokens)

            pairwise_tokens_collated.append(pairwise_tokens)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            attn_maps_collated.append((unary_attn, pairwise_attn))

        pairwise_tokens_collated = torch.cat(pairwise_tokens_collated)
        logits = self.box_pair_predictor(pairwise_tokens_collated)

        return logits, prior_collated, \
            boxes_h_collated, boxes_o_collated, object_class_collated, attn_maps_collated
