import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

# --- 1. Hyperparameters & Device ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Transformation Pipeline ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((7, 7), interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.Pad((0, 0, 1, 1), fill=0)
])

# --- 3. Load Datasets ---
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Data loaded successfully.")

# --- 4. Physical Architecture ---
class OptoelectronicClassifier(nn.Module):
    def __init__(self):
        super(OptoelectronicClassifier, self).__init__()
        self.flatten = nn.Flatten()

        # 1st Optical MVM: 64 LEDs project to 64 Photodiodes
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(32))

        # 2nd Optical MVM: 32 hidden LEDs project to 10 Output Photodiodes
        self.readout = nn.Linear(32, 10, bias=False)
        self.bias2 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = self.flatten(x)

        # Enforce non-negative light transmission without killing gradients (Dead ReLU fix)
        w1_pos = torch.abs(self.fc1.weight)
        out1 = F.linear(x, w1_pos)

        # Route the 64 photodiode signals into 32 positive and 32 negative pairs
        pos_signals = out1[:, :32]
        neg_signals = out1[:, 32:]

        # Physical differential amplifier & LED forward-bias ReLU (WITH ELECTRONIC BIAS)
        diff = pos_signals - neg_signals + self.bias1
        hidden_acts = F.relu(diff)

        # Second mask non-negative transmission
        w2_pos = torch.abs(self.readout.weight)
        out2 = F.linear(hidden_acts, w2_pos)
        
        # Final electronic addition of the readout bias
        out_logits = out2 + self.bias2

        return out_logits

# --- 5. Initialize Model, Loss, and Optimizer ---
# Note: Re-initializing everything here prevents the NameError
model = OptoelectronicClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Model architecture compiled and moved to device.")

# --- 6. Training & Testing Loops ---
def train_model():
    print("Starting Training...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

def test_model():
    print("\nStarting Testing...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the 10,000 test images: {accuracy:.2f}%")

# --- 7. Execution ---
train_model()
test_model()

# --- 8. Export ---
weights_filename = 'mnist_64x64_weights.pth'
torch.save(model.state_dict(), weights_filename)
print(f"\nModel weights saved as {weights_filename}.")