import torch


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
        return out_spikes


class Network:
    def __init__(
        self,
        num_in: int,
        num_out: int,
    ):
        self.layers = [Layer(100, 100) for _ in range(3)]

        self.layers[0] = Layer(num_in, 100)
        self.layers[-1] = Layer(100, num_out)

    def forward(
        self,
        input_spikes: torch.Tensor,
    ):
        spikes = input_spikes

        for layer in self.layers:
            spikes = layer.forward(spikes)

        return spikes


n_in = 100
n_out = 100

network = Network(n_in, n_out)

in_tensor = torch.rand(n_in).round()
out_tensor = network.forward(in_tensor)

# print(f"In: {in_tensor}, sum: {in_tensor.sum()}")
# print(f"Out: {out_tensor}, sum: {out_tensor.sum()}")


differences = []

for i in range(10000):    
    in_tensor = torch.rand(n_in).round()
    out_tensor = network.forward(in_tensor)
    
    difference = (in_tensor.sum() - out_tensor.sum()).item()
    
    differences.append(difference)


from collections import Counter
import matplotlib.pyplot as plt

frequency = Counter(differences)

sorted_items = sorted(frequency.items())
x_values, y_values = zip(*sorted_items) if sorted_items else ([], [])

plt.figure(figsize=(10, 6))
plt.bar(x_values, y_values, color='skyblue', edgecolor='black', width=1.0)

plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Differences')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(x_values)
plt.tight_layout()

plt.show()