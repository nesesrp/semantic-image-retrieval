import torch

embeddings = torch.load("outputs/embeddings_1000.pt")

print("Total number of images:", len(embeddings))
print()

for name, tensor in list(embeddings.items())[:5]:
    print(f"{name} -> shape: {tensor.shape}")
    print(tensor[:10])  # Display the first 10 values (all 512 values would not fit on the screen)
    print("-" * 40)