# 🧠 Multimodal Glaucoma Detection (FusionNet + JointStreamNet)

Code-focused repository for multimodal glaucoma detection using retinal fundus images.

This project implements and evaluates:
- **FusionNet** — feature-level multimodal fusion
- **JointStreamNet** — parallel streams with late fusion

## 🚀 Start

```bash
git clone https://github.com/syma-afsha/Multimodal_Glaucoma_Detection.git
cd Multimodal_Glaucoma_Detection

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install common dependencies
pip install numpy pandas matplotlib scikit-learn opencv-python pillow tqdm
pip install torch torchvision
