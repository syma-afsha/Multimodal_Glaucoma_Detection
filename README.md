# 👁️ Multimodal Glaucoma Detection (FusionNet + JointStreamNet)

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

## 📁 Repository Structure

Multimodal_Glaucoma_Detection/
├── main_fixed.ipynb # Main training + evaluation notebook (recommended)
├── main.ipynb # Alternative/original notebook
├── draw.py # Plotting / utility script
│
├── plots/ # Training curves + comparison plots (AUC/ACC/Loss)
├── train_grids/ # Saved grids / visual outputs from training
├── samples_final/ # Sample outputs (final model)
├── samples_joint/ # Sample outputs (joint stream model)
├── samples_pairs/ # Sample outputs (paired samples)
│
├── cm.png # Confusion matrix (example)
├── cm_final.png # Confusion matrix (final)
├── output.png # Example output visualization
├── predict.png # Example prediction visualization
├── pred.png # Example prediction visualization
├── compare.png # Model comparison figure
├── compare_final.png # Final model comparison figure
│
├── augmented_glaucoma/ # Generated augmented data (ignored)
├── augmented_glaucoma_only/ # Generated augmented data (ignored)
├── augmented_multimodal/ # Generated augmented data (ignored)
├── glaucoma_imgs/ # Local image folder (ignored if large)
├── glaucoma_cropped_aug/ # Cropped/augmented images (ignored)
├── synthetic_glaucoma/ # Synthetic dataset outputs (ignored)
│
├── checkpoints_final/ # Saved checkpoints (ignored)
├── checkpoints_joint/ # Saved checkpoints (ignored)
├── checkpoints_clinical/ # Saved checkpoints (ignored)
│


