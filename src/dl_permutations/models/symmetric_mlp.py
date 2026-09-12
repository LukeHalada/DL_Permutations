import torch

class MLP(torch.nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes, activations, bias=True, dtype=torch.double):
        super(MLP, self).__init__()
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.activations = activations
        self.bias = bias

        self.encoder = torch.nn.ModuleList(
            [torch.nn.Linear(self.encoder_sizes[i - 1], self.encoder_sizes[i], bias=bias, dtype=dtype)
             for i in range(1, len(self.encoder_sizes))]
        )
 
        self.decoder = torch.nn.ModuleList(
            [torch.nn.Linear(self.encoder_sizes[-1], self.decoder_sizes[0], bias=bias, dtype=dtype)]
        )
        self.decoder.extend([
            torch.nn.Linear(self.decoder_sizes[i - 1], self.decoder_sizes[i], bias=bias, dtype=dtype)
            for i in range(1, len(self.decoder_sizes))
        ])

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def encode(self, x):
        x_inv = x.flip(0).flatten(start_dim=1)
        x = x.flatten(start_dim=1)
        for i in range(len(self.encoder)-1):
            x = self.encoder[i](x)
            x = self.activations[i](x)

            x_inv = self.encoder[i](x_inv)
            x_inv = self.activations[i](x_inv)

        return (x + x_inv)/2
        
    def decode(self, emb):
        for i in range(len(self.decoder)):
            emb = self.decoder[i](emb)
            emb = self.activations[i+len(self.encoder)](emb)

        return emb