import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
from utils import nst_model
from torch.autograd import Variable
from torch.optim import LBFGS
import cv2



# from models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def image_loading(data_dir, content_img_name, style_img_name, max_size):

    # get content_img-path
    content_images_dir = os.path.join(data_dir, 'content_images')
    content_img_path = os.path.join(content_images_dir, content_img_name)
    content_image = cv2.imread(content_img_path)

    # get style image path
    style_images_dir = os.path.join(data_dir, 'style_images')
    style_img_path = os.path.join(style_images_dir, style_img_name)
    style_image = cv2.imread(style_img_path)


    # Resize content image
    content_height, content_width, _ = content_image.shape
    print(f"Content Image Size Before Resizing: {content_image.shape[0]}x{content_image.shape[1]}")

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

    # Optional: Display resized image dimensions
    print(f"Content Image Size After Resizing: {content_image.shape[0]}x{content_image.shape[1]}")

    # Convert images to RGB for visualization
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)

    return content_image, style_image


def preprocess_image(image, device):
    # Convert the image to float32 and normalize to [0, 1] range
    image = image.astype(np.float32) / 255.0

    # Normalize using ImageNet's mean and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),  # Scale to [0, 255] range
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    
    # Apply the transformations and move the tensor to the specified device
    image_tensor = transform(image).to(device).unsqueeze(0)  # Add batch dimension

    return image_tensor

def visualise_final_image(tensor):
    # Move the tensor to the CPU, detach it, and remove the batch dimension
    tensor = tensor.detach().cpu().squeeze(0)

    # Denormalize the tensor
    mean = torch.tensor(IMAGENET_MEAN_255).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD_NEUTRAL).view(3, 1, 1)
    tensor = tensor * std + mean  # Reverse the normalization
    
    # If the tensor shape is [C, H, W], convert it to [H, W, C] (required for imshow)
    if tensor.shape[0] == 3:  # Assuming RGB image
        tensor = tensor.permute(1, 2, 0)

    # Convert to numpy and clip values to [0, 255], then convert to uint8 for imshow
    image = tensor.numpy().clip(0, 255).astype(np.uint8)

    return image


def prepare_model(device):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    model = nst_model.Vgg19(requires_grad=False, show_progress=True)

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def neural_style_transfer(config, content_img, style_img, device):

    num_of_iterations = 1000

    # clone the content image which will then be optimised
    init_img = content_img
    optimizing_img = Variable(init_img, requires_grad=True)


    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    # line_search_fn does not seem to have significant impact on result
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            if cnt % 25 == 0:
                print(f'L-BFGS | iteration: {cnt:03}/{num_of_iterations}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            # utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

        cnt += 1
        return total_loss

    optimizer.step(closure)

    return optimizing_img