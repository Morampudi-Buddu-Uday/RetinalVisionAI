📌 README.md – RetinalVisionAI: Deep Learning-Based Retinal Disease Detection
>
🧠 Overview
RetinalVisionAI is a deep learning-based web app that automates the detection of two major causes of vision loss:

Diabetic Retinopathy (DR)

Glaucoma

Using pre-trained CNN models and a simple UI built with Streamlit, users can upload retinal fundus images and get real-time predictions — making early detection more accessible and scalable.

🔬 Motivation
"Preventable blindness should never be a matter of resources or access."

Traditional screening methods like fundus examination or OCT are effective but costly, time-consuming, and require specialists. This project leverages AI to:

Enable early-stage screening

Reduce ophthalmologist workload

Promote healthcare equity

🏗️ Architecture & Models
Three powerful CNN architectures were explored for performance comparison:

🌀 EfficientNet – high performance with low computational cost

🧱 ResNet-50 – deep residual learning to solve vanishing gradients

📐 VGG16 – simple and effective for spatial feature extraction

Final implementation uses EfficientNet, fine-tuned on labeled retinal datasets for:

DR classification (5 stages)

Glaucoma detection (binary)

🖼️ Demo Preview
Upload a retinal image

Get predictions:

DR: One of 5 severity stages

Glaucoma: Healthy or Glaucomatous

Animated visual feedback

Medical recommendations for DR based on result

🚀 Getting Started
🔧 Requirements
Install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
📁 Project Structure
bash
Copy
Edit
RetinalVisionAI/
├── Animations/           # Lottie animations for UI
├── Models/               # .h5 trained model files (tracked via Git LFS)
├── Scripts/              # Individual DR and Glaucoma test scripts
├── Test_Images/          # Sample fundus images for testing
├── final_test.py         # Main Streamlit UI combining models
├── requirements.txt
└── README.md
🧪 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/Morampudi-Buddu-Uday/RetinalVisionAI.git
cd RetinalVisionAI
Run the Streamlit app:

bash
Copy
Edit
streamlit run final_test.py
🗂️ Sample Output
Diabetic Retinopathy: Moderate
"Closer monitoring every 3–6 months, laser therapy advised."

Glaucoma: Detected
"🚨 Signs of Glaucoma detected. Consult an ophthalmologist."

📦 Dependencies
streamlit

streamlit-lottie

tensorflow

numpy

pillow

matplotlib

scikit-learn

(See requirements.txt)

⚖️ License
This project is released under the MIT License.

🤝 Contributions
Feel free to fork the repo and contribute! Pull requests are welcome.
