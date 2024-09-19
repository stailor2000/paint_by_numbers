import streamlit as st
import torch
import platform
from PIL import Image
import numpy as np
import cv2
import utils.utils_nst as utils_nst
import utils.utils as utils
import zipfile
import io

TABS_STYLE_Y1 = ['Uploaded Content Image', 'Style Image', 'Added Style Adaption']
TABS_STYLE_Y2 = ['Image to be Converted', 'Paint by Numbers - Coloured Rendering', 'Paint by Numbers Canvas' ]
TABS_STYLE_N = ['Uploaded Image', 'Paint by Numbers - Coloured Rendering', 'Paint by Numbers Canvas' ]

st.title('Paint By Numbers Generator')


def check_device():
    """Check and display the device being used"""
    if platform.system() == "Darwin":  # checking if it's macOS
        if torch.backends.mps.is_available():
            return torch.device("mps")
    elif torch.cuda.is_available():  # if not macOS, check for CUDA availability
        return torch.device("cuda")
    else:  # if neither macOS nor CUDA, default to CPU
        return torch.device("cpu")
  

def download_images(images, labels):
    bytes_buffer = io.BytesIO()
    with zipfile.ZipFile(bytes_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img, label in zip(images, labels):
            img_byte_arr = io.BytesIO()
            # Check if the image is a numpy array and convert to PIL Image if necessary
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            # Now, save the PIL Image to a bytes array
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            zipf.writestr(label, img_byte_arr)
    bytes_buffer.seek(0)
    zip_bytes = bytes_buffer.getvalue()

    # Button to download zip file
    st.download_button(
        label="Download Files",
        data=zip_bytes,
        file_name="images.zip",
        mime="application/zip"
    )


def preprocess_nst(content_image, style_image):
    max_size = 800

     # Resize content image
    content_height, content_width, _ = content_image.shape

    if max(content_height, content_width) > max_size:
        scale_factor = max_size / float(max(content_height, content_width))
        new_height = int(content_height * scale_factor)
        new_width = int(content_width * scale_factor)
        content_image = cv2.resize(content_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Resize style image
    style_height, style_width, _ = style_image.shape
    if max(style_height, style_width) > max_size:
        scale_factor = max_size / float(max(style_height, style_width))
        new_height = int(style_height * scale_factor)
        new_width = int(style_width * scale_factor)
        style_image = cv2.resize(style_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return content_image, style_image


def preprocess_generator(image):
    max_dimension = 2000
    min_dimension = 1000

    height, width, _ = image.shape
    if height > max_dimension or width > max_dimension:
        scaling_factor = max_dimension / max(height, width)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height))
    elif height < min_dimension or width < min_dimension:
        scaling_factor = min_dimension / min(height, width)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image)

    return(image)

def quantisation(image, paint_config):
    with st.spinner('Generating Image... This could take a couple of minutes.'):
        # Apply SLIC superpixel segmentation
        segments = utils.slic_superpixel_segmentation(image, n_segments=paint_config['n_segments'], compactness=paint_config['compactness'])
    
        # Quantize superpixels to a limited color palette
        quantized_image, quantized_indices, quantized_colors = utils.quantize_superpixels(image, segments, n_colors=paint_config['number_colours'])
    
        # Display the quantized image
        st.image(quantized_image)

        return quantized_image, quantized_indices, quantized_colors
    

def create_bw_canvas(quantized_indices, quantized_colors):
    with st.spinner('Generating Image... This could take a couple of minutes.'):
        # Create edges and labeled image
        edges = utils.create_paint_by_number_edges(quantized_indices)
        labeled_image_on_white = utils.label_regions_within_edges(quantized_indices, edges)

        # Add a black border around 'labeled_image_on_white'
        labeled_image_on_white_with_border = utils.add_black_border(labeled_image_on_white, border_size=1)

        # Get the quantised colour palette
        palette_image = utils.display_color_palette(quantized_colors, colors_per_row=10)

        # Display the color palette
        st.image(labeled_image_on_white_with_border)
        st.image(palette_image)

        return labeled_image_on_white_with_border, palette_image
        


def only_generator(uploaded_file, paint_config):
    tabs = st.tabs(TABS_STYLE_N)

    with tabs[0]:       # Uploaded Image
        if uploaded_file is not None:
            # Open the image with PIL and convert to numpy array
            pil_image = Image.open(uploaded_file)
            uploaded_image = np.array(pil_image)

            # Preprocess the image
            image = preprocess_generator(uploaded_image)
        else:
            st.write("No file uploaded yet. Please upload an image to display.")

    with tabs[1]:       # Paint by Numbers Coloured Rendering
        if uploaded_file is not None:
            # Quantise the image and display it
            quantized_image, quantized_indices, quantized_colors = quantisation(image, paint_config)

    with tabs[2]:
        if uploaded_file is not None:
            # Create paint by numbers canvas and colour palette
            canvas_image, palette_image = create_bw_canvas(quantized_indices, quantized_colors)

    if uploaded_file is not None:
        images = [image, quantized_image, canvas_image, palette_image]
        labels = ['original_image.jpeg', 'coloured_painting.jpeg', 'paint_by_numbers.jpeg', 'colour_palette.jpeg']

        # Create a zip file of all images
        download_images(images, labels)


        


def apply_adaption(uploaded_file, uploaded_file_style, nst_config, paint_config, device):
    tabs1 = st.tabs(TABS_STYLE_Y1)
    tabs2 = st.tabs(TABS_STYLE_Y2)

    with tabs1[0]:       # Content image
        if uploaded_file is not None and uploaded_file_style is not None:
            # Open the content image with PIL and convert to numpy array
            pil_image_content = Image.open(uploaded_file)
            uploaded_image_content = np.array(pil_image_content)

            # Open the style image with PIL and convert to numpy array
            pil_image_style = Image.open(uploaded_file_style)
            uploaded_image_style = np.array(pil_image_style)

            # Preprocess images
            content_image, style_image = preprocess_nst(uploaded_image_content, uploaded_image_style)

            # Display content image
            st.image(content_image)

        elif uploaded_file is not None and uploaded_file_style is None:
            st.write("Please upload a style image.")
        elif uploaded_file is None and uploaded_file_style is not None:
            st.write("Please upload an image to be adapted.")
    with tabs1[1]: # Style image
        if uploaded_file is not None and uploaded_file_style is not None:

            # Display style image
            st.image(style_image)

    with tabs1[2]:  # Adapted Image
        if uploaded_file is not None and uploaded_file_style is not None:
            # Convert image np arrays to tensors
            content_tensor = utils_nst.preprocess_image(content_image, device)
            style_tensor = utils_nst.preprocess_image(style_image, device)

            # Setup the progress bar placeholder
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0, text="Operation in progress.... Could take a few minutes.")
            
            def update_progress(current_step, total_steps):
                progress = current_step / total_steps
                if progress <= 1.0:
                    progress_bar.progress(progress, text="Operation in progress.... Could take a few minutes.")
                else:
                    # Once complete, remove the progress bar
                    progress_placeholder.empty()  # This removes the widget

            # Run style adaption training with progress update
            merged_image_tensor = utils_nst.neural_style_transfer(nst_config, content_tensor, style_tensor, device, progress_update=update_progress)

            # Convert to numpy array 
            merged_image_np = utils_nst.visualise_final_image(merged_image_tensor)

            # Display adapted image
            st.image(merged_image_np)

    with tabs2[0]:  # Adapted Image
        if uploaded_file is not None and uploaded_file_style is not None:
            # Preprocess the image
            image = preprocess_generator(merged_image_np)

    with tabs2[1]:       # Paint by Numbers Coloured Rendering
        if uploaded_file is not None and uploaded_file_style is not None:
            # Quantise the image and display it
            quantized_image, quantized_indices, quantized_colors = quantisation(image, paint_config)

    with tabs2[2]:
        if uploaded_file is not None and uploaded_file_style is not None:
            # Create paint by numbers canvas and colour palette
            canvas_image, palette_image = create_bw_canvas(quantized_indices, quantized_colors)
        
    # download button
    if uploaded_file is not None and uploaded_file_style is not None:
        images = [image, quantized_image, canvas_image, palette_image]
        labels = ['original_image.jpeg', 'coloured_painting.jpeg', 'paint_by_numbers.jpeg', 'colour_palette.jpeg']

        # Create a zip file of all images
        download_images(images, labels)



if __name__ == "__main__":

    # Using the sidebar for user inputs
    with st.sidebar:
        st.header('User Inputs')

        device = check_device()
        st.sidebar.write(f"Device: {device}")  # Display the device info in the sidebar

        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

        # Option to choose style adaptation
        apply_style = st.radio("Apply Style Transfer?", ('No', 'Yes'))

        # Conditional inputs based on the choice to apply style
        if apply_style == 'Yes':
            st.subheader("Style Transfer Options")
            # style_factor = st.slider("Style Intensity", 0, 100, 50)
            # style_type = st.selectbox("Choose Style Type", ['Monet', 'Van Gogh', 'Pop Art'])
            uploaded_file_style = st.file_uploader("Choose a style image", type=['png', 'jpg', 'jpeg'])
            nst_config = {
            'style_weight': st.slider('Style Weight', min_value=1e0, max_value=1e5, value=3e4),
            'tv_weight': st.slider('Total Variation Weight', min_value=1e0, max_value=1e6, value=1e0),
            'content_weight': 1e5
        }
            num_tabs1 = len(TABS_STYLE_Y1) 
            num_tabs2 = len(TABS_STYLE_Y2) 
        else:
            num_tabs = len(TABS_STYLE_N) 

        
        # variables for paint by numbers
        st.subheader("Paint by Number Style Options")
        paint_config = {
            'number_colours': st.slider('Number of Colours', min_value=1, max_value=30, value=20),
            'n_segments': st.slider('Number of Segments', min_value=1000, max_value=5000, value=2000),
            'compactness': st.slider('Compactness (Square-ness)', min_value=5, max_value=20, value=10)  # Adjusted max_value for slider range
        }

    # Main screen display


    if apply_style == 'Yes':
        apply_adaption(uploaded_file, uploaded_file_style, nst_config, paint_config, device)
    else:
        only_generator(uploaded_file, paint_config)

