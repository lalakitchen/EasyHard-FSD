# [IEEE-TAI 2026] EasyHard-FSD  
**Learning From Easy to Hard: Fingerprint Spoof Detection with Hard Sample Mining**

---

## 📢 Paper Status  
**Accepted for publication** in *IEEE Transactions on Artificial Intelligence (IEEE-TAI)*.  
**To appear.**

---
## 👥 Authors  

**Wenny Ramadha Putri***, **Farchan Raswa Hakim***, Bach-Tung Pham, Shang-Kuan Chen, Chung-I Huang, Kuo-Chen Li, Shih-Lun Chen, **Jia-Ching Wang†**

\* Co-first authors  † Corresponding author



## 📌 Overview  

This repository contains the official implementation of **EasyHard-FSD**, a fingerprint presentation attack detection (PAD) framework that progressively learns from **easy samples to hard samples**.

The proposed method integrates:

- Loss-based **Hard Sample Mining (HMM)**
- **Teacher–Student learning**
- **Knowledge Distillation (KD)**
- **Exponential Moving Average (EMA)** teacher update

The framework is designed for **robust fingerprint spoof detection**, **cross-scanner generalization**, and **reproducible biometric security research**.

---

## ✨ Key Features  

- Baseline fingerprint PAD training  
- Hard sample mining (η-based)  
- Teacher–student training loop  
- Knowledge distillation with temperature scaling  
- EMA-stabilized teacher update  
- ACE evaluation (FAR / FRR)  
- Grad-CAM heatmap visualization  
- Clean experimental separation for fair comparison  

---

## 📂 Repository Structure  

.
├── train_baseline.py        # Baseline training (no hard mining, no KD, no EMA)  
├── train_hardsample.py      # Baseline + hard sample mining only  
├── train_our.py             # EasyHard-FSD (HMM + KD + EMA teacher)  
├── evaluate.py              # ACE evaluation (FAR / FRR)  
├── heatmap.py               # Grad-CAM visualization  
├── utils/  
│   └── ImageGenerator.py    # Data utilities  
├── checkpoint/              # Saved model checkpoints  
├── Visualizations_hard/     # Grad-CAM outputs  
└── README.md  

---

## 🧠 Method Summary  

### 1. Baseline Training  
A fingerprint PAD model is trained using all available training samples without curriculum learning.

### 2. Hard Sample Mining (HMM)  
Samples are ranked by classification loss, and the top **η% hardest samples** are selected for focused learning.

### 3. Teacher–Student Learning  
- A **teacher model** is trained on the full dataset.  
- A **student model** is fine-tuned on the mined hard samples.

### 4. Knowledge Distillation (KD)  
The student is optimized using a combination of:
- Cross-entropy loss with ground-truth labels  
- KL divergence with softened teacher predictions  

### 5. EMA Teacher Update  
After student optimization, the teacher is updated using:
θ_teacher ← α · θ_teacher + (1 − α) · θ_student

This EMA update stabilizes training and mitigates noise accumulation.

---

## 🚀 Usage  

### Baseline Training  
python train_baseline.py --year 2015 --scanner Digital_Persona --exp_name baseline

### Hard Sample Mining Only  
python train_hardsample.py --year 2015 --scanner Digital_Persona --exp_name hardsample

### EasyHard-FSD (Proposed Method)  
python train_our.py --year 2015 --scanner Digital_Persona --exp_name easyhard

### Evaluation (ACE)  
python evaluate.py --year 2015 --scanner Digital_Persona --method our --exp_name easyhard

### Grad-CAM Visualization  
python heatmap.py --year 2015 --scanner Digital_Persona --method our --exp_name easyhard

---

## 📊 Evaluation Metric  

ACE = (FAR + FRR) / 2

where FAR is the False Acceptance Rate and FRR is the False Rejection Rate.

---

## 📦 Dataset  

This work uses the **LivDet (Liveness Detection) fingerprint datasets**, which are publicly available.

Dataset download:
https://livdet.org/

Please follow the dataset license and usage terms provided by the LivDet organizers.

---



## 📜 License  

This project is released for **academic research purposes only**.  
Please contact the corresponding author for commercial usage.

---

## 📬 Contact  
**Farchan Hakim Raswa:**
E-mail: farchan.hakim.r@g.ncu.edu.tw
