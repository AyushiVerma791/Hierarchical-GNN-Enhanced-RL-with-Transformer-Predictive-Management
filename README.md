# Hierarchical GNN-Enhanced RL with Transformer-Based Predictive Management

An intelligent, proactive liquidity management framework for Blockchain Payment Channel Networks (PCNs) like the Lightning Network. This project utilizes a dual-agent Reinforcement Learning architecture, temporal traffic prediction, and graph attention mechanisms to prevent transaction failures caused by channel depletion.

## Objective
Design and implement an AI-powered routing and rebalancing system. Network liquidity is managed using a hybrid Deep Learning approach:
*   **Temporal Prediction:** A PyTorch `TransformerEncoder` predicts future transaction traffic and channel depletion risks.
*   **Topological Encoding:** A Multi-Head Graph Attention Network (`GATConv`) dynamically identifies and weights highly congested network hubs.
*   **Hierarchical Routing:** A dual-agent Proximal Policy Optimization (PPO) system decouples global liquidity budgeting (High-Level Agent) from local multi-path routing (Low-Level Agent).

## Key Outcomes
*   Integrate Graph Neural Networks (GNNs) with sequence-predicting Transformers in a PyTorch environment.
*   Solve the "action-space explosion" problem in RL by using a hierarchical actor-critic architecture.
*   Simulate Barabási-Albert scale-free graph topologies to accurately mimic real-world blockchain hubs.
*   Visualize AI decision-making and network bottlenecks in real-time using an interactive Streamlit web dashboard.

## Tools & Technologies
*   **Frontend / Dashboard:** Streamlit, Matplotlib
*   **Machine Learning:** Python, PyTorch, PyTorch Geometric (PyG)
*   **Reinforcement Learning:** Proximal Policy Optimization (PPO)
*   **Network Simulation:** NetworkX, NumPy

## Prerequisites
*   Python 3.10+
*   pip (Python package manager)

## Setup & Run

**1) Clone the repository**
```bash
git clone https://github.com/AyushiVerma791/Hierarchical-GNN-Enhanced-RL-with-Transformer-Predictive-Management.git
cd Hierarchical-GNN-Enhanced-RL-with-Transformer-Predictive-Management
```

**2) Install dependencies**
```bash
pip install -r requirements.txt
```

**3) Launch the Dashboard (Recommended)**
```bash
streamlit run app.py
# Open http://localhost:8501 in your browser
```

**4) Run Offline Training (Optional)**
```bash
# To retrain the PPO agents from scratch
python train.py --config config.yaml
```

## How to Test the Secure Rebalancing Flow
**1. View Baseline Network**
Select "Stage 1" on the dashboard. The app generates a healthy Barabási-Albert scale-free network where central hubs hold maximum liquidity.

**2. Inject High Traffic (The Problem)**
Select "Stage 2". The system simulates exponential transaction traffic. You will see central hub channels turn red, indicating severe liquidity depletion and a dropping transaction success ratio.

**3. Execute AI Rebalancing (The Solution)**
Select "Stage 3". The HGRL-TPM model activates. The activity log will display the High-Level agent selecting a regional budget and the Low-Level agent routing circular transactions. Red channels are restored to optimal liquidity.

## Project Structure
```text
HGRL-TPM/
│
├── shared_env/                 # Network simulation & math
│   ├── pcn_env.py              # Barabási-Albert graph environment
│   └── transaction_generator.py# Exponential traffic simulator
│
├── models/                     # PyTorch Architectures
│   ├── gat_encoder.py          # Graph Attention Network (GAT)
│   ├── transformer_predictor.py# Sequence prediction model
│   └── hierarchical_agents.py  # Dual-PPO Actor-Critic agents
│
├── eval/                       # Saved pre-trained model weights (.pth)
│
├── app.py                      # Streamlit interactive dashboard
├── train.py                    # Main offline RL training loop
├── config.yaml                 # Topology scale & hyperparameter config
├── requirements.txt            
└── README.md
```


