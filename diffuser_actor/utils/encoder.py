import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from .position_encodings import RotaryPositionEncoding3D
from .layers import FFWRelativeCrossAttentionModule, ParallelAttention
from .resnet import load_resnet50, load_resnet18, replace_submodules, ResNet18_custom
from .clip import load_clip

NUM_CAMERAS=4
DP_ENCODER="resnet18"

class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_sampling_level=3,
                 nhist=3,
                 num_attn_heads=8,
                 num_vis_ins_attn_layers=2,
                 fps_subsampling_factor=5):
        super().__init__()
        assert backbone in ["resnet50", "resnet18", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor

        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim
        )
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid
            # at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Instruction encoder
        self.instruction_encoder = nn.Linear(512, embedding_dim)

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim, n_heads=num_attn_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])

        ######################################################################
        # Dict for storing new key models, including 2D, 2.5D, 3D

        # Uncomment these for enabling 2D encoders for each camera
        # key_model_map=nn.ModuleDict()
        # key_transform_map=nn.ModuleDict()

        # #2D model for 2D  images only, each camera has its own encoder
        # for i in range(NUM_CAMERAS):
        #     model,key_transform_map[f"2D_encoder_{i}"]=load_resnet18(pretrained=False)
        #     key_model_map[f"2D_encoder_{i}"]=replace_submodules(
        #         root_module=model,
        #         predicate=lambda x: isinstance(x,nn.BatchNorm2d),
        #         func=lambda x: nn.GroupNorm(
        #             num_groups=x.num_features//16,
        #             num_channels=x.num_features)
        #         )
        
        # self.key_model_map=key_model_map
        # self.key_transform_map=key_transform_map

        # When using 2D features only, 3D  point cloud features can either be learned 
        # from rgb images or be dummy point cloud without useful information
        self.rgb_to_pcd = nn.Sequential(
            nn.Linear(3,3),
            nn.ReLU(),
            nn.Linear(3,3)
        )

        self.encoder_3d=ResNet18_custom(out_channels=embedding_dim,spatial_softmax_size=(32,32,512),pretrained=True)

            

    def forward(self):
        return None

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, nhist, 3+)

        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
                                    context_feats, context)

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
            goal_gripper[:, None], self.goal_gripper_embed,
            context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
            - gripper_pos: (B, nhist, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos

    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]
            pcd_i = F.interpolate(
                pcd,
                (feat_h, feat_w),
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3,
            device=instruction.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats

######################################################################
    # Add 2D encoder for each camera
    # def encode_images_2d(self,rgb):
    #     """
    #     Computer 2D visual features using different encoder for different camera view,
    #     batch norm replaced with group norm already,
    #     (?)Spatial softmax pooling is used to get the final feature instead of global average pooling,
        
    #     (!)Without concatenate features together.

    #     Args:
    #         - rgb: (B, ncam, 3, H, W), pixel intensities

    #     Returns:
    #         - rgb_feats: [(B, ncam, F)]
    #     """
    #     assert rgb.shape[1]==NUM_CAMERAS

    #     rgb_features=[]
    #     for i in range(NUM_CAMERAS):
    #         images=rgb[:,i,...] # of shape (B,3,H,W)
    #         curr_encoder=self.key_model_map[f"2D_encoder_{i}"]
    #         curr_normalizer=self.key_transform_map[f"2D_encoder_{i}"]
    #         normalized_images=curr_normalizer(images)
    #         curr_features=curr_encoder(normalized_images) # of shape (B,1000)
    #         rgb_features.append(curr_features)
    #     rgb_features=torch.stack(rgb_features,dim=1) # of shape(B,ncam,1000)

    #     return rgb_features

    # TODO: integrate these image encoding methods to a single function
    def encode_2D_images(self, rgb):
        """
        Compute visual features embeddings only at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
        """

        # Generate point cloud from rgb images
        rgb=einops.rearrange(rgb,"bt ncam c h w -> bt ncam h w c")
        pcd=self.rgb_to_pcd(rgb)
        pcd=einops.rearrange(pcd,"bt ncam h w c -> bt ncam c h w")
        rgb=einops.rearrange(rgb,"bt ncam h w c -> bt ncam c h w")

        return self.encode_images(rgb,pcd)

    def encode_3d_pd(self,pcd):
        """
        Compute 3D point cloud features using 3D encoder

        Args:
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - context_feats: (B, ncam, F)
        """
        num_camera=pcd.shape[1]
        pcd=einops.rearrange(pcd,"bt ncam c h w -> (bt ncam) c h w")
        context_feats=self.encoder_3d(pcd)  # of shape (B*ncam,32,32,embedding_dim)
        feat_h,feat_w=context_feats.shape[-3:-1]
        context=F.interpolate(
            pcd,
            (feat_h,feat_w),
            mode='bilinear'
        )
        context_feats=einops.rearrange(context_feats,"(bt ncam) h w c -> bt (ncam h w) c ",ncam=num_camera)
        context=einops.rearrange(context,"(bt ncam) c h w -> bt (ncam h w ) c ",ncam=num_camera)
        return context_feats,context