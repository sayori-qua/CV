import torch
import torchvision
import cv2
import numpy as np
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)
def get_masks(frame, threshold=0.68):
    transform = T.ToTensor()
    img_tensor = transform(frame)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model([img_tensor])
    masks = output[0]['masks']
    scores = output[0]['scores']
    labels = output[0]['labels']
    high_confidence_indices = scores > threshold
    masks = masks[high_confidence_indices]
    labels = labels[high_confidence_indices]
    binary_masks = (masks > 0.5).squeeze(1).detach().cpu().numpy()

    return binary_masks, labels #маски для найденных объектов и лейблы

def get_color_from_id(class_id, max_classes=80):
    np.random.seed(class_id % max_classes)
    color = tuple(np.random.randint(50, 255, size=3).tolist())
    np.random.seed()
    return color

def apply_masks_to_frame(frame, binary_masks, labels, alpha=0.5):
    overlay = frame.copy()
    class_names = {
        1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane',
        6: 'Bus', 7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic light',
        11: 'Fire hydrant', 13: 'Stop sign', 14: 'Parking meter', 15: 'Bench',
        16: 'Bird', 17: 'Cat', 18: 'Dog', 19: 'Horse', 20: 'Sheep',
        21: 'Cow', 22: 'Elephant', 23: 'Bear', 24: 'Zebra', 25: 'Giraffe',
        27: 'Backpack', 28: 'Umbrella', 31: 'Handbag', 32: 'Tie', 33: 'Suitcase',
        34: 'Frisbee', 35: 'Skis', 36: 'Snowboard', 37: 'Sports ball', 38: 'Kite',
        39: 'Baseball bat', 40: 'Baseball glove', 41: 'Skateboard', 42: 'Surfboard',
        43: 'Tennis racket', 44: 'Bottle', 46: 'Wine glass', 47: 'Cup', 48: 'Fork',
        49: 'Knife', 50: 'Spoon', 51: 'Bowl', 52: 'Banana', 53: 'Apple',
        54: 'Sandwich', 55: 'Orange', 56: 'Broccoli', 57: 'Carrot', 58: 'Hot dog',
        59: 'Pizza', 60: 'Donut', 61: 'Cake', 62: 'Chair', 63: 'Couch',
        64: 'Potted plant', 65: 'Bed', 67: 'Dining table', 70: 'Toilet',
        72: 'TV', 73: 'Laptop', 74: 'Mouse', 75: 'Remote', 76: 'Keyboard',
        77: 'Cell phone', 78: 'Microwave', 79: 'Oven', 80: 'Toaster'
    }
    min_mask_area = 600
    for idx, (mask, label) in enumerate(zip(binary_masks, labels)):
        if np.sum(mask) < min_mask_area:
            continue
        class_id = int(label)
        class_name = class_names.get(class_id, f'Unknown ({class_id})')
        color = get_color_from_id(class_id)
        overlay[mask] = color
        cv2.putText(frame, class_name, (10, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame
