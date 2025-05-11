# QMazeRL: Exploring Quantum Reinforcement Learning for Maze Solving

Welcome to **QMazeRL**, a research-focused project exploring the application of **Quantum Reinforcement Learning (QRL)** in solving maze navigation problems. This repository provides a comparative study between traditional heuristic methods, classical Deep Q-Learning agents, and hybrid quantum-classical agents built using parameterized quantum circuits.

---

## 🧠 Motive

The project investigates how principles of quantum computing—specifically quantum circuits—can be used to enhance reinforcement learning agents. Mazes are a classical benchmark for decision-making and planning algorithms, and QMazeRL uses them to:

- Analyze QRL performance under structured constraints,
- Compare it with classical and heuristic baselines,
- Evaluate generalization to dynamically generated maze layouts.

---

## 🛠️ Installation

### 📋 Requirements

- Python 3.8+
- PennyLane
- TensorFlow
- NumPy
- Matplotlib

### 💻 Setup Steps

1. **Clone the repository**

   ```bash
   git clone
   cd QMazeRL
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train or test agents**
   ```bash
   python train_agent.py
   ```

---

## 📊 Results

### 🔍 Evaluation Summary

| Approach          | Max Maze Size Solved | Avg. Steps to Goal | Adaptability |
| ----------------- | -------------------- | ------------------ | ------------ |
| Heuristic (DFS)   | 10x10                | Low                | Low          |
| Classical DQN     | 15x15                | Moderate           | Moderate     |
| Quantum DQN (QRL) | 12x12                | Moderate           | High         |

### 🖼️ Visualized Results

- **Heuristic (DFS)**  
  Deterministic and fast but fails in dynamic layouts.  
  ![Heuristic Maze](./results/heuristic_maze.png)

- **Classical Reinforcement Learning (DQN)**  
  Learns with experience; performance improves over time.  
  ![Classical DQN Maze](./results/classical_dqn.png)

- **Quantum DQN (QRL)**  
  Integrates quantum circuits with neural networks to learn navigation policies. Promising results in maze generalization.  
  ![Quantum DQN Maze](./results/qrl_maze.png)

---

## ⚙️ Methodology

### Heuristic (DFS)

A rule-based depth-first search algorithm used for solving mazes with known structure. It is non-learning and deterministic.

### Classical Reinforcement Learning (DQN)

- Uses Deep Q-Networks to estimate Q-values.
- Learns from experience replay and Bellman updates.

### Quantum Reinforcement Learning (QRL)

- Integrates parameterized quantum circuits via PennyLane.
- Hybrid model merges quantum features into classical layers.
- Q-values are predicted using quantum-enhanced states.

---

## 🔮 Future Work

1. **Quantum Circuit Optimization** – Refine circuit architectures for better state encoding.
2. **Transfer Learning** – Apply trained QRL models to novel maze designs.
3. **Hybrid Architectures** – Explore combining classical and quantum policies.

---

## 🤝 Contributions

QMazeRL is a collaborative project focused on the potential of quantum computation in real-world RL applications. Contributions are welcome!

---

## 📬 Contact

For feedback, suggestions, or collaborations, please reach out via GitHub Issues or Pull Requests.

Enjoy exploring Quantum RL with **QMazeRL**! 🚀
