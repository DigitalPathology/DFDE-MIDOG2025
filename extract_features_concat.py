import torch
import torch.nn as nn
import os
import argparse
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from torchvision.models.resnet import Bottleneck, ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Preprocessing ===
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


# === Barlow Twins Model Loader ===
class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.mean(dim=[2, 3])  # Global Average Pooling


def get_barlow_twins_model():
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3])

    # 🟩 Replace with your actual local path to the model
    local_model_path = "D:/dl/MIDOG_2025_Track_2/bt_rn50_ep200.torch"

    print(f"Loading Barlow Twins weights from: {local_model_path}")
    state_dict = torch.load(local_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# === Feature Extraction Function ===
def extract_and_concat_features(image_path, hibou_model, bt_model, output_path):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        hibou_feat = hibou_model(image_tensor).pooler_output.cpu()
        bt_feat = bt_model(image_tensor).cpu()
        fused_feat = torch.cat([hibou_feat, bt_feat], dim=1)
    torch.save(fused_feat, output_path)
    print(f"Saved concatenated features to {output_path}")


# === Main ===
parser = argparse.ArgumentParser(description='Extract and Concatenate HIBOU-L + Barlow Twins Features')
parser.add_argument('--image_dir', type=str, required=True, help='Directory containing PNG images')
parser.add_argument('--feat_dir', type=str, required=True, help='Directory to save features')
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(args.feat_dir, exist_ok=True)

    print("Loading HIBOU-L model...")
    hibou_model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True).to(device)

    print("Loading Barlow Twins model...")
    bt_model = get_barlow_twins_model()

    for filename in os.listdir(args.image_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(args.image_dir, filename)
            output_file = os.path.join(args.feat_dir, filename.replace('.png', '_hibou_bt.pt'))
            print(f"Processing: {filename}")
            extract_and_concat_features(image_path, hibou_model, bt_model, output_file)
