class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Number of channels
        c0 = 3
        c1 = 32
        c2 = 64
        c3 = 64
        c4 = 32
        c5 = 2

        # Kernel size
        k = 4
        k0 = k
        k1 = k
        k2 = k
        k3 = k
        k4 = k

        # Stride
        s = 2
        s0 = s
        s1 = s
        s2 = s
        s3 = s
        s4 = s

        # Formula for transposed conv: (input - 1) * stride - 2 * padding + kernel + output_padding
        # add the ", padding= 1, or output_padding=1" to the convolutions when needed

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=c0, out_channels=c1, kernel_size=k0,stride=s0, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k1,stride=s1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=k2,stride=s2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=c3,out_channels=c4, kernel_size=k3,stride=s3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=c4,out_channels=c5, kernel_size=k4,stride=s4, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c5, out_channels=c4, kernel_size=k4, stride=s4, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(in_channels=c4, out_channels=c3, kernel_size=k3, stride=s3, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(in_channels=c3, out_channels=c2, kernel_size=k2, stride=s2, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(in_channels=c2, out_channels=c1, kernel_size=k2, stride=s1, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose2d(in_channels=c1, out_channels=c0, kernel_size=k1, stride=s0, padding=1, output_padding=0),
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self,x):
        return self.encoder(x)

    def decode(self,x):
        return self.decoder(x)

net = Net().to(device)
summary(net, image_shape)
