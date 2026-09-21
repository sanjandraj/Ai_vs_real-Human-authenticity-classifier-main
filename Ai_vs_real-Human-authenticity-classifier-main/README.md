# Ai_vs_real-Human-authenticity-classifier  

## **Overview**  
A deep learning-based classifier to distinguish **real human faces** from **AI-generated** ones using **ResNet50**.  

## **Dataset** 
- **9.6K images** from Kaggle:  
  - **Real**: 5000 human faces  
  - **AI-Generated**: 4630 synthetic faces  
- Split: **Train (70%)**, **Test (15%)**, **Validation (15%)**  

## **Model**   
- **Pretrained ResNet50** with custom layers:  
  - **GlobalAveragePooling2D**, **Dense (128, ReLU)**, **Dense (1, Sigmoid)**  
- **Trained for 15 epochs**, fine-tuned for better accuracy.  

## **Deployment (Streamlit App)**   
- Upload an image and classify it as **Real** or **AI-Generated**  
- Uses **final_model.h5** for predictions  

## **Run Locally** 
```bash
pip install tensorflow streamlit pillow numpy  
streamlit run app.py  
```  

## **Future Work** 
🔹 Improve accuracy with more data  
🔹 Add explainability (Grad-CAM)  
🔹 Deploy online  

**Contributions welcome!** 
