import time
import torch
import base64
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from resnet import resnet18, resnet34, resnet50

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.jpeg')
    
def load_model(architecture):
    if architecture == 'ResNet-18':
        return resnet18(pretrained=True)
    elif architecture == 'ResNet-34':  
        return resnet34(pretrained=True)
    elif architecture == 'ResNet-50':  
        return resnet50(pretrained=True)

# Define a transform to resize the image to 32x32 and normalize it.
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))  
])

def side_show():
    """Shows the sidebar components for the Image Thresholding and returns user inputs as dict."""
    with st.sidebar:
        st.write("#### Select ResNet Architecture")
        option = st.selectbox("options", ['ResNet-18', 'ResNet-34','ResNet-50'])
        st.write('###  Selected:', option)
    return option

def main():
    st.title("CIFAR-10 ResNet Image Classifier")
    architecture = side_show()
    uploaded_image = st.file_uploader("Upload a CIFAR-10 image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        model = load_model(architecture)
        model.eval()

        # Preprocess the uploaded image
        image = Image.open(uploaded_image)
        processed_image = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(processed_image)

        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        predicted_class = class_names[output.argmax()]

        # Display results
        st.image(image, caption="Uploaded Image", use_column_width=True)

        
        if st.button("Classify"):
             with st.spinner('Predicting class...'):
                time.sleep(1)  # Add a 1-second delay
                st.write("#### Predicted Class:", predicted_class)
        
if __name__ == "__main__":
    main()
