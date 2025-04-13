# Explainable AI for Finger Movement Estimation from sEMG & Biomechanics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<br/>

**Developing an explainable artificial intelligent (XAI) model to estimate finger movement from the integration of surface electromyography signals and biomechanics of the hand**

---

## 1. Overview

This repository contains the research and implementation for the thesis project focused on developing a novel system for estimating finger joint movements using surface electromyography (sEMG) signals. The core innovation lies in integrating advanced signal processing, biomechanical modeling, and **Explainable Artificial Intelligence (XAI)**.

The primary goal is to create a system that not only achieves high accuracy in predicting finger kinematics but also provides transparent, physiologically interpretable insights into *how* the model arrives at its predictions by relating sEMG patterns to specific muscle activations and movements. This work aims to bridge the gap between complex AI models and the need for understanding the underlying neuromuscular control mechanisms.

Potential applications include advanced prosthetic control, targeted neurorehabilitation strategies, and more intuitive human-computer interaction (HCI) systems.

**Keywords:** Surface Electromyography (sEMG), Finger Movement Estimation, Explainable AI (XAI), Machine Learning, Deep Learning, Biomechanics, Signal Processing, Neural Networks.

---

## 2. Project Goals & Objectives

The overarching goal is **to develop and validate a novel, explainable system for accurately estimating finger joint movements from sEMG signals, incorporating advanced signal processing and AI techniques, and providing insights into the physiological aspects of muscle control and finger kinematics.**

### Specific Objectives:

1.  **Advanced sEMG Signal Processing:**
    * Investigate and implement techniques (e.g., wavelet transform, empirical mode decomposition, common spatial patterns) to enhance sEMG quality, reduce noise/artifacts, and extract relevant features.
    * Optimize the processing pipeline for real-time performance.
    * Evaluate processing effectiveness (SNR, feature separability).
2.  **Explainable AI Model Development:**
    * Develop and train ML/DL models (e.g., attention-based neural networks, hybrid CNNs/RNNs) for accurate sEMG-to-joint-angle mapping.
    * Integrate XAI mechanisms (e.g., attention weights visualization, SHAP, LIME) to interpret model behavior.
    * Optimize model architecture for accuracy and real-time capability.
3.  **Physiological Interpretation:**
    * Utilize XAI outputs to analyze individual muscle/group contributions to specific movements.
    * Investigate spatiotemporal muscle activation patterns and their link to finger kinematics.
    * Relate model explanations to existing physiological knowledge and biomechanical principles.
4.  **Validation and Evaluation:**
    * Utilize open-source kinematic datasets to drive a mathematical model of the finger skeleton.
    * Develop a stimulation model to generate synthetic sEMG data corresponding to skeleton movements.
    * Evaluate the AI model using synthetic data via Leave-One-Subject-Out (LOSO) cross-validation.
    * Validate physiological interpretations against the stimulation model and known principles.
    * Assess overall system performance (accuracy, speed, interpretation quality) and compare against existing methods.

---

## 3. Features

* **Sophisticated Signal Processing:** Employs state-of-the-art techniques (Wavelets, EMD, CSP) for robust sEMG feature extraction.
* **Cutting-Edge AI Models:** Leverages attention mechanisms and hybrid deep learning architectures (CNNs, RNNs) tailored for time-series sEMG data.
* **Built-in Explainability:** Integrates established XAI methods (SHAP, LIME, Attention Maps) for model transparency and interpretation.
* **Biomechanical Simulation:** Incorporates a mathematical finger skeleton model and a muscle activation stimulation model for generating realistic synthetic data.
* **Rigorous Validation Framework:** Uses LOSO cross-validation on synthetic data for robust generalization assessment.
* **Comparative Analysis:** Benchmarks performance against established finger movement estimation techniques.
* **Focus on Physiological Insight:** Aims to derive meaningful understanding of neuromuscular control strategies from model explanations.

---

## 4. Methodology

The project follows a structured approach:

1.  **Literature Review:** Comprehensive review of sEMG processing, XAI, biomechanical hand models, and finger kinematics estimation.
2.  **Data Strategy:**
    * Leverage existing open-source kinematic datasets (e.g., motion capture data of finger movements).
    * Develop a mathematical model representing the finger's skeletal structure and kinematics based on this data.
    * Create a stimulation model to simulate realistic muscle activation patterns (sEMG) required to produce the movements defined by the skeleton model. This generates paired synthetic sEMG and joint angle data.
3.  **Signal Processing Pipeline:** Implement and optimize selected algorithms (Wavelet, EMD, etc.) to clean raw synthetic sEMG and extract relevant temporal and spectral features.
    * *Example:* Denoising using wavelet thresholding, followed by feature extraction (e.g., RMS, Mean Frequency, Wavelet Coefficients).
4.  **AI Model Development:** Design, implement, and train explainable AI models (Attention-based NN, CNN-RNN hybrids) using the processed synthetic sEMG features as input and corresponding joint angles as output.
    * *Example:* An LSTM network with an attention layer to focus on relevant sEMG time segments, trained using Mean Squared Error loss.
5.  **XAI Integration:** Apply techniques like SHAP or attention map visualization to understand feature importance and model decision-making for specific movements.
    * *Example:* Generating SHAP value plots to show the contribution of each sEMG channel feature to a specific predicted joint angle.
6.  **Testing & Validation:**
    * Perform LOSO cross-validation using the synthetic dataset.
    * Evaluate performance using metrics like Root Mean Square Error (RMSE) for joint angles, correlation coefficients, and processing latency.
    * Compare results against baseline models or published methods.
    * Validate the physiological interpretations derived from XAI against the known parameters of the stimulation model and established biomechanics.
7.  **Analysis & Interpretation:** Analyze quantitative results and qualitative XAI outputs to draw conclusions about model performance and physiological insights.

---

## 5. Theoretical Background

* **Surface Electromyography (sEMG):** Non-invasive technique to record electrical activity produced by skeletal muscles. Signals are complex, non-stationary, and reflect the neuromuscular activation underlying movement.
* **Machine Learning/Deep Learning:** Algorithms used to learn patterns from data. Relevant models include Recurrent Neural Networks (RNNs, LSTMs) for time-series data, Convolutional Neural Networks (CNNs) for spatial/temporal feature extraction, and Attention Mechanisms to focus on relevant input parts.
* **Explainable AI (XAI):** Methods designed to make AI model decisions understandable to humans. Techniques like LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) provide insights into feature contributions for specific predictions. Attention weights in models can also indicate input importance.
* **Biomechanics of the Hand:** Study of the mechanical forces and structures involved in hand and finger movement, including skeletal kinematics and muscle actuation principles.

---

## 6. Project Structure




## 7. Getting Started

### Prerequisites

* **Python:** Version 3.8 or higher recommended.
* **Package Manager:** `pip` (usually included with Python) or `conda`.
* **Git:** For cloning the repository.
* **Key Python Libraries:** (See `requirements.txt` for full list)
    * `numpy`: Fundamental package for numerical computation.
    * `scipy`: For scientific and technical computing (signal processing).
    * `pandas`: Data manipulation and analysis.
    * `scikit-learn`: Machine learning tools (preprocessing, metrics).
    * `tensorflow` or `pytorch`: Deep learning frameworks.
    * `matplotlib`, `seaborn`: Plotting and visualization.
    * `shap` / `lime`: XAI libraries (if used).
    * `jupyterlab` / `notebook`: For running notebooks.
* **(Optional) MATLAB:** If parts of the simulation or processing use MATLAB scripts.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git) # Replace with your actual repository URL
    cd your-repo-name
    ```

2.  **Set up Virtual Environment (Recommended):**
    * Using `venv`:
        ```bash
        python -m venv venv
        source venv/bin/activate  # Linux/macOS
        # venv\Scripts\activate  # Windows
        ```
    * Using `conda`:
        ```bash
        conda create -n xai_semg python=3.9 # Or desired version
        conda activate xai_semg
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Setup:**
    * Download required open-source kinematic datasets (provide specific instructions/links here).
    * Run scripts to generate synthetic data using the simulation models (provide commands).
        ```bash
        # Example command (replace with actual script)
        # python src/generate_synthetic_data.py --kinematic_data_path data/raw/ --output_path data/processed/
        ```

---

## 8. Usage Examples

*(Note: Replace script names and arguments with your actual implementation)*

1.  **Data Preprocessing (Synthetic Data):**
    ```bash
    # Assuming synthetic data is already generated
    python src/signal_processing.py --input_dir data/processed/synthetic_raw_semg --output_dir data/processed/synthetic_processed_semg --config config/processing_config.yaml
    ```

2.  **Train an AI Model:**
    ```bash
    python src/train.py --data_dir data/processed/ --features_file synthetic_processed_semg.csv --labels_file joint_angles.csv --model_type attention_lstm --output_dir results/trained_models/ --log_dir results/logs/ --config config/training_config.yaml
    ```

3.  **Evaluate Model Performance:**
    ```bash
    python src/evaluate.py --model_path results/trained_models/attention_lstm_best.h5 --test_data_dir data/processed/test_set/ --output_dir results/metrics/ --config config/evaluation_config.yaml
    ```

4.  **Generate Explanations (XAI):**
    ```bash
    python src/explainability.py --model_path results/trained_models/attention_lstm_best.h5 --data_sample data/processed/sample_for_explanation.csv --method shap --output_dir results/figures/explanations/
    ```

5.  **Run Jupyter Notebooks:**
    ```bash
    jupyter lab # or jupyter notebook
    # Navigate to the 'notebooks/' directory and open desired notebooks
    ```

---

## 9. Validation and Results

* **Performance Metrics:** The primary metrics for evaluation include:
    * **Accuracy:** Root Mean Square Error (RMSE) and Correlation Coefficient (CC) between predicted and actual joint angles.
    * **Real-time Capability:** Average processing time per sample/window.
    * **Explainability Quality:** Qualitative assessment of XAI outputs (e.g., consistency, physiological plausibility) and potentially quantitative metrics if applicable (e.g., faithfulness).
* **Validation Strategy:** Leave-One-Subject-Out (LOSO) cross-validation on the synthetically generated dataset ensures the model generalizes across different simulated subjects.
* **Comparison:** Performance is compared against baseline methods (e.g., standard regression models on basic features) and potentially other published sEMG-based estimation techniques.
* **Results Storage:** Detailed metrics, figures, and model outputs are stored in the `results/` directory. Key findings will be summarized in the thesis document.

---

## 10. Explainability for Physiological Insight

A core component of this project is leveraging XAI to understand the underlying physiological processes:

* **Muscle Contribution Analysis:** Using methods like SHAP or attention weights to identify which sEMG channels (representing muscles or muscle groups) contribute most significantly to the prediction of specific joint movements (e.g., index finger flexion).
* **Spatiotemporal Patterns:** Visualizing attention maps or time-varying feature importance to understand how muscle activation patterns evolve over time during a movement sequence.
* **Validation of Insights:** Comparing the patterns revealed by XAI with the known muscle functions from the stimulation model and established physiological literature to confirm the validity of the interpretations.

---

## 11. Contributing

Contributions to this research project are welcome. Please follow these guidelines:

1.  **Fork** the repository.
2.  Create a **new branch** for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes, ensuring code is well-commented and follows existing style.
4.  Add **unit tests** for new functionality.
5.  Ensure all tests pass.
6.  Commit your changes (`git commit -m 'Add some feature'`).
7.  Push to the branch (`git push origin feature/your-feature-name`).
8.  Open a **Pull Request** with a clear description of the changes.

Please report any bugs or issues using the GitHub Issues tracker.

---

## 12. License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

---

## 13. Acknowledgments

* **Supervisors:** Tran Le Giang, Ph.D., Assoc. Prof. Le Ngoc Bich, Ph.D.
* **Institution:** School of Biomedical Engineering, International University, Vietnam National Universities-Ho Chi Minh City.
* **Data Sources:** Acknowledge the providers of any open-source kinematic datasets used.
* **Software:** Acknowledge key libraries (TensorFlow/PyTorch, Scikit-learn, SHAP, etc.).

---

## 14. Contact

* **Tạ Minh Trí**
    * Email: `bebeiu21284@student.hcmiu.edu.vn`
    * GitHub: https://github.com/taminhtri2003/
* **Project Repository:** https://github.com/taminhtri2003/Finger-Estimation-Using-surface-EMG

---

