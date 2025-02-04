import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        down = self.pool(self.conv(x))
        skip_connection = self.conv(x)

        expexted_size = tuple(s // 2 for s in x.shape[2:])
        return down, skip_connection

class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            skip_connection: Skip connection tensor of shape (B, C_skip, H, W)
        Returns:
            Output tensor of shape (B, C_out, H*2, W*2)
        """
        x = self.up(x)
        # If needed, pad x to match skip_connection size
        if x.shape != skip_connection.shape:
            diffY = skip_connection.size(2) - x.size(2)
            diffX = skip_connection.size(3) - x.size(3)
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 11, disable_skip_connections: bool = False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            disable_skip_connections: Whether to set all the skip connections to zero.
        """
        super(UNet, self).__init__()
        self.disable_skip_connections = disable_skip_connections

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up2 = Up(64, 128, 32)
        self.up1 = Up(32, 64, 11)

        self.final_conv = nn.Conv2d(11, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        input_shape = x.shape[2:]
        down1, skip1 = self.down1(x)
        down2, skip2 = self.down2(down1)

        bottleneck = self.bottleneck(down2)

        up2 = self.up2(bottleneck, skip2 if not self.disable_skip_connections else torch.zeros_like(skip2))
        up1 = self.up1(up2, skip1 if not self.disable_skip_connections else torch.zeros_like(skip1))

        output = self.final_conv(up1)
        return output