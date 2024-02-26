import torch.nn as nn

from morphospaces.networks._components.unet_blocks import (
    DoubleConv,
    ResNetBlock,
    create_decoders,
    create_encoders,
    number_of_features_per_level,
)


class BaseUNet3D(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation
            mask.             It's up to the user of the class to interpret
            the out_channels and use the proper loss criterion during training
            (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each
            level of the encoder;
            if it's an integer the number of feature maps is given
            by the geometric progression: f_maps ^ k, k=1,2,3,4
        basic_module: basic model for the encoder/decoder
            (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in
            `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path
            (applied only if f_maps is an int) default: 4
        conv_kernel_size (int or tuple): size of the convolving
            kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all
            three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(
                f_maps, num_levels=num_levels
            )
        self.f_maps = f_maps

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert (
                num_groups is not None
            ), "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            pool_kernel_size,
        )

        # create decoder path
        self.decoders = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
        )

        # in the last layer a 1×1 convolution reduces the number
        # of output channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


class UNet3D(BaseUNet3D):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor
    upsampling in the decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        conv_padding=1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            conv_padding=conv_padding,
        )


class MultiscaleUnet3D(BaseUNet3D):
    def __init__(
        self,
        in_channels,
        out_channels,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        conv_padding=1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            conv_padding=conv_padding,
        )

        decoder_final_convs = []
        for index in range(num_levels - 1):
            decoder_index = (num_levels - 2) - index
            decoder_final_convs.append(
                nn.Conv3d(self.f_maps[decoder_index], out_channels, 1)
            )
        self.decoder_final_convs = nn.ModuleList(decoder_final_convs)

    def training_forward(self, x):
        """Perform inference and return both the result and
        the intermediate decoder outputs.
        """
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        decoder_outputs = []
        for decoder, decoder_final_conv, encoder_features in zip(
            self.decoders, self.decoder_final_convs, encoders_features
        ):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            decoder_outputs.append(decoder_final_conv(x))

        x = self.final_conv(x)

        return x, decoder_outputs

    def forward(self, x):
        """Perform inference and return only the result."""
        return self.training_forward(x)[0]


class ResidualUNet3D(BaseUNet3D):
    """
    Residual 3DUnet model implementation based on
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions
    for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net,
    in theory it allows for deeper UNet.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        conv_padding=1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            conv_padding=conv_padding,
        )
