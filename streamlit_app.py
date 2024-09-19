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

TABS_STYLE_Y = ['Uploaded Content Image', 'Style Image', 'Added Style Adaption', 'Paint by Numbers - Coloured Rendering', 'Paint by Numbers Canvas' ]
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
    
# def preprocess_nst():


def create_zip_from_images(images, labels):
    """Create a ZIP archive from a list of images.
    
    Args:
    images (list): List of images where each can be either a PIL.Image or a numpy.ndarray.
    labels (list of str): Corresponding labels for the files in the archive.
    
    Returns:
    bytes: The byte content of the ZIP archive.
    """
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
    return bytes_buffer.getvalue()


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

        # Display the color palette
        palette_image = utils.display_color_palette(quantized_colors, colors_per_row=10)

        st.image(labeled_image_on_white_with_border)
        st.image(palette_image)

        return labeled_image_on_white_with_border, palette_image
        



def only_generator(uploaded_file, number_colours):
    tabs = st.tabs(TABS_STYLE_N)

    with tabs[0]:       # Uploaded Image
        if uploaded_file is not None:
            # Open the image with PIL and convert to numpy array
            pil_image = Image.open(uploaded_file)
            uploaded_image = np.array(pil_image)

            image = preprocess_generator(uploaded_image)
            # image = Image.fromarray(image)      # convert to PIL image
        else:
            st.write("No file uploaded yet. Please upload an image to display.")

    with tabs[1]:       # Paint by Numbers Coloured Rendering
        if uploaded_file is not None:
            quantized_image, quantized_indices, quantized_colors = quantisation(image, number_colours)
            # quantized_image = Image.fromarray(quantized_image)      # convert to PIL image

    with tabs[2]:
        if uploaded_file is not None:
            canvas_image, palette_image = create_bw_canvas(quantized_indices, quantized_colors)
            # canvas_image = Image.fromarray(canvas_image)      # convert to PIL image

    if uploaded_file is not None:
        images = [image, quantized_image, canvas_image, palette_image]
        labels = ['original_image.jpg', 'coloured_painting.jpg', 'paint_by_numbers.jpg', 'colour_palette.jpg']

        # buf = io.BytesIO()
        # with zipfile.ZipFile(buf, "x") as myzip: # set the mode parameter to x to create and write a new file
        #     myzip.writestr('original_image.jpg', image)
        #     myzip.writestr('coloured_painting.jpg', quantized_image) 
        #     myzip.writestr('paint_by_numbers.jpg', canvas_image)
        #     myzip.writestr('colour_palette.jpg', palette_image)


        # # with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # # for img, label in zip(images, labels):
        # #     img_byte_arr = io.BytesIO()
        # #     # Check if the image is a numpy array and convert to PIL Image if necessary
        # #     if isinstance(img, np.ndarray):
        # #         img = Image.fromarray(img)
        # #     # Now, save the PIL Image to a bytes array
        # #     img.save(img_byte_arr, format='PNG')
        # #     img_byte_arr = img_byte_arr.getvalue()
        # #     zipf.writestr(label, img_byte_arr)

        # st.download_button(
        #     label="Download zip file",
        #     data=buf.getvalue(),
        #     file_name="zip file.zip",
        #     mime="data/zip"
        # )

        zip_bytes = create_zip_from_images(images, labels)

        # if st.button('Download Images as ZIP'):
        #     zip_bytes = create_zip_from_images(images, labels)
        st.download_button(
            label="Download Files",
            data=zip_bytes,
            file_name="images.zip",
            mime="application/zip"
        )

        


def apply_adaption():
    tabs = st.tabs(TABS_STYLE_Y)


if __name__ == "__main__":

    # Using the sidebar for user inputs
    with st.sidebar:
        st.header('User Inputs')

        device = check_device()
        st.sidebar.write(f"Device: {device}")  # Display the device info in the sidebar

        uploaded_file = st.file_uploader("Choose a photo file", type=['png', 'jpg', 'jpeg'])

        # Option to choose style adaptation
        apply_style = st.radio("Apply Style Transfer?", ('No', 'Yes'))

        # Conditional inputs based on the choice to apply style
        if apply_style == 'Yes':
            st.subheader("Style Transfer Options")
            style_factor = st.slider("Style Intensity", 0, 100, 50)
            # style_type = st.selectbox("Choose Style Type", ['Monet', 'Van Gogh', 'Pop Art'])
            num_tabs = len(TABS_STYLE_Y)  # Five tabs if style transfer is applied
        else:
            num_tabs = len(TABS_STYLE_N)  # Three tabs otherwise

        
        # variables for paint by numbers
        st.subheader("Paint by Number Style Options")
        paint_config = {
            'number_colours': st.slider('Number of Colours', min_value=1, max_value=30, value=20),
            'n_segments': st.slider('Number of Segments', min_value=1000, max_value=5000, value=2000),
            'compactness': st.slider('Compactness (Square-ness)', min_value=5, max_value=20, value=10)  # Adjusted max_value for slider range
        }

    # Main screen display


    if apply_style == 'Yes':
        apply_adaption()
    else:
        only_generator(uploaded_file, paint_config)

