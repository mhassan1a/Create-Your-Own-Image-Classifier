import argparse
import torch
import json
from torch import nn
from torchvision import models, transforms
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Predict the flower name from an image')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    image_tensor = process_image(args.input)
    top_probs, top_classes = predict(image_tensor, model, args.top_k)

    for i in range(len(top_probs)):
        flower_class = top_classes[i]
        flower_name = cat_to_name[str(flower_class)]
        prob = top_probs[i]
        print(f"{flower_name}: {prob:.4f}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    resize = transforms.Resize(256)
    image = resize(image)

    # Center-crop the resized image
    crop = transforms.CenterCrop(224)
    image = crop(image)

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Normalize the image
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using
                             std=[0.229, 0.224, 0.225])    # ImageNet stats
    ])
    normalized_image = normalize_transform(image_array)
    res = normalized_image.numpy()
    #print(res)
    return res   # Return as a NumPy array


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
  
    # Load the image and preprocess it
    image = Image.open(image_path)
    processed_image = process_image(image)
    
    # Convert the NumPy array to a PyTorch tensor
    tensor_image = torch.from_numpy(processed_image)
    
    # Add a batch dimension and move tensor to the appropriate device (CPU or GPU)
    tensor_image = tensor_image.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_image = tensor_image.to(device, dtype=torch.float)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(tensor_image)
        prob = torch.exp(output)
        #print(len(prob[0]))
        top_probs, top_classes = prob.topk(topk, dim=1)
    
    #print(top_classes)
    top_classes = [cat_to_name[str(class_n.item()+1)] for class_n in top_classes[0]]
    
    return top_probs[0].cpu().numpy(), top_classes


if __name__ == '__main__':
    main()
