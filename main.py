import torch


class Neuron:
    def __init__(
        self,
        num_out: int,
    ):
        self.w = torch.randn(num_out)

    def fire(
        self,
    ):
        probs = torch.softmax(self.w, dim=-1)
        spikes = torch.bernoulli(probs)
        return spikes


neuron = Neuron(100)
out = neuron.fire()
print(torch.sum(out))


class Layer:
    def __init__(
        self,
        num_in: int,
        num_out: int,
        device: str = "cuda",
    ):
        self.num_in = num_in
        self.num_out = num_out

        self.w = torch.randn(num_in, num_out).to(device)

    def forward(
        self,
        in_spikes: torch.Tensor,
    ):
        assert (
            in_spikes.dim() == 1
        ), f"Expected 1D tensor, got {in_spikes.dim()}D tensor"
        assert (
            in_spikes.size(-1) == self.num_in
        ), f"Expected tensor of size [*, {self.num_in}], got [*, {in_spikes.size(1)}]"

        in_spikes = in_spikes.to(self.w)

        in_spikes.unsqueeze_(-1)
        to_fire = torch.broadcast_to(in_spikes, (self.num_in, self.num_out))
        probs = torch.softmax(self.w, dim=-1) * to_fire
        out_spikes = (torch.sum(torch.bernoulli(probs), dim=0) != 0).float()
        print(out_spikes)


layer = Layer(3, 5)
bar = torch.Tensor([1, 0, 1])
layer.forward(bar)
