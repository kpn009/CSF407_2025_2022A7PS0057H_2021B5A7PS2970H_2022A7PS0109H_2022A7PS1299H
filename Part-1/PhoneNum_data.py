import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Set fixed seed for reproducibility globally
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Worker init function that sets seed for worker processes
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# Generator for DataLoader reproducibility
g = torch.Generator()
g.manual_seed(SEED)

class PhoneNumberDataset(Dataset):
    """
    Dataset for generating phone number images.
    """
    def __init__(self, num_samples=20000, is_balanced=True, transform=None, save_dir='../src/Dataset/'):
        """
        Args:
            num_samples: Number of samples to generate
            is_balanced: Whether to generate balanced or imbalanced dataset
            transform: Optional transforms to apply
            save_dir: Directory to save the dataset
        """
        self.num_samples = num_samples
        self.is_balanced = is_balanced
        self.transform = transform
        self.save_dir = save_dir
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Country codes for data augmentation
        self.country_codes = ['01', '44', '91', '86', '81', '49', '33', '61', '55', '82']
        
        # Generate phone numbers and corresponding images
        self.data = self._generate_data()
    
    def _generate_phone_number(self, is_balanced=True):
        """
        Generate a random 10-digit phone number.
        
        For imbalanced datasets, we ensure that up to 4 digits are repeated frequently 
        across different phone numbers.
        """
        if is_balanced:
            # Completely random 10-digit number
            return ''.join(random.choices('0123456789', k=10))
        else:
            # For imbalanced data: use 4 repeated digits at random positions
            repeated_digits = random.choices('0123456789', k=4)
            other_digits = random.choices('0123456789', k=6)
            
            # Mix the repeated and other digits
            positions = random.sample(range(10), 4)  # Positions for repeated digits
            phone_number = [''] * 10
            
            digit_idx = 0
            for pos in positions:
                phone_number[pos] = repeated_digits[digit_idx]
                digit_idx += 1
            
            # Fill remaining positions
            other_idx = 0
            for i in range(10):
                if phone_number[i] == '':
                    phone_number[i] = other_digits[other_idx]
                    other_idx += 1
            
            return ''.join(phone_number)
    
    def _create_image(self, phone_number, add_country_code=False):
        """
        Create an image of the phone number with optional country code
        """
        # Create a blank image (28x280 for 10 digits) with padding
        image_width = 280
        image_height = 28
        image = Image.new('L', (image_width, image_height), color=255)
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to use a default font
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Add country code if needed
        if add_country_code:
            country_code = random.choice(self.country_codes)
            phone_number = country_code + phone_number
        
        # Draw each digit with clear spacing
        for i, digit in enumerate(phone_number):
            # Calculate position (evenly spaced)
            x_position = i * (image_width // len(phone_number)) + 5
            # Draw digit in black (0) on white background (255)
            draw.text((x_position, 2), digit, font=font, fill=0)
        
        # Convert to numpy array and normalize properly
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add channel dimension for compatibility with transforms
        img_tensor = torch.tensor(img_array).unsqueeze(0)  # Shape becomes [1, 28, 280]
        
        return img_tensor
    
    def _generate_data(self):
        """
        Generate phone number datasets
        """
        data = []
        for _ in range(self.num_samples):
            # Generate phone number
            phone_number = self._generate_phone_number(self.is_balanced)
            
            # Create image
            # 50% chance to add country code for data augmentation
            add_country_code = random.random() > 0.5
            image = self._create_image(phone_number, add_country_code)
            
            # Store phone number and image
            data.append({
                'phone_number': phone_number,
                'image': image,
                'label': torch.tensor([int(digit) for digit in phone_number], dtype=torch.long)
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            # Apply transformations if provided
            image = self.transform(image)
        
        return image, label
    
    def save_dataset(self, filename):
        """
        Save the dataset to disk
        """
        torch.save(self.data, os.path.join(self.save_dir, filename))
        print(f"Dataset saved to {os.path.join(self.save_dir, filename)}")
    
    @classmethod
    def load_dataset(cls, filepath, transform=None):
        """
        Load the dataset from disk
        """
        data = torch.load(filepath)
        dataset = cls(num_samples=1, transform=transform)  # Create a minimal instance
        dataset.data = data  # Replace with loaded data
        dataset.num_samples = len(data)
        return dataset

    def visualize_samples(self, num_samples=5):
        """
        Visualize some random samples from the dataset
        """
        indices = random.sample(range(len(self.data)), min(num_samples, len(self.data)))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i, idx in enumerate(indices):
            img = self.data[idx]['image'].squeeze(0).numpy()  # Remove channel dimension
            label = ''.join([str(l.item()) for l in self.data[idx]['label']])
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Phone: {label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def get_data_loaders(balanced_path=None, imbalanced_path=None, batch_size=64, train_ratio=0.7, val_ratio=0.15, 
                     create_new=True, num_samples=20000, save_dir='../src/Dataset/'):
    """
    Get data loaders for training, validation, and testing.
    If paths are provided, load datasets from files, otherwise create new datasets.
    
    Args:
        balanced_path: Path to balanced dataset file
        imbalanced_path: Path to imbalanced dataset file
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        create_new: Whether to create new datasets or load from paths
        num_samples: Number of samples to generate for new datasets
        save_dir: Directory to save/load datasets
    
    Returns:
        Dictionary containing data loaders for balanced and imbalanced datasets
    """
    # Define transforms with proper normalization
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])
    
    if create_new:
        # Create balanced dataset
        balanced_dataset = PhoneNumberDataset(
            num_samples=num_samples,
            is_balanced=True,
            transform=transform,
            save_dir=save_dir
        )
        balanced_filename = "phone_balanced.pt"
        balanced_dataset.save_dataset(balanced_filename)
        balanced_path = os.path.join(save_dir, balanced_filename)
        
        # Create imbalanced dataset
        imbalanced_dataset = PhoneNumberDataset(
            num_samples=num_samples,
            is_balanced=False,
            transform=transform,
            save_dir=save_dir
        )
        imbalanced_filename = "phone_imbalanced.pt"
        imbalanced_dataset.save_dataset(imbalanced_filename)
        imbalanced_path = os.path.join(save_dir, imbalanced_filename)
    else:
        # Load datasets from files
        balanced_dataset = PhoneNumberDataset.load_dataset(balanced_path, transform=transform)
        imbalanced_dataset = PhoneNumberDataset.load_dataset(imbalanced_path, transform=transform)
    
    # Split datasets into train/val/test sets
    def split_indices(dataset_size):
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        train_end = int(dataset_size * train_ratio)
        val_end = train_end + int(dataset_size * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return train_indices, val_indices, test_indices
    
    # Split balanced dataset
    b_train_idx, b_val_idx, b_test_idx = split_indices(len(balanced_dataset))
    
    # Split imbalanced dataset
    i_train_idx, i_val_idx, i_test_idx = split_indices(len(imbalanced_dataset))
    
    # Create subsets
    b_train_subset = Subset(balanced_dataset, b_train_idx)
    b_val_subset = Subset(balanced_dataset, b_val_idx)
    b_test_subset = Subset(balanced_dataset, b_test_idx)
    
    i_train_subset = Subset(imbalanced_dataset, i_train_idx)
    i_val_subset = Subset(imbalanced_dataset, i_val_idx)
    i_test_subset = Subset(imbalanced_dataset, i_test_idx)
    
    # Create data loaders with fixed generator instead of worker_init_fn
    loaders = {
        'balanced': {
            'train': DataLoader(b_train_subset, batch_size=batch_size, shuffle=True, 
                               worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(b_val_subset, batch_size=batch_size, shuffle=False,
                             worker_init_fn=seed_worker, generator=g),
            'test': DataLoader(b_test_subset, batch_size=batch_size, shuffle=False,
                              worker_init_fn=seed_worker, generator=g)
        },
        'imbalanced': {
            'train': DataLoader(i_train_subset, batch_size=batch_size, shuffle=True,
                               worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(i_val_subset, batch_size=batch_size, shuffle=False,
                             worker_init_fn=seed_worker, generator=g),
            'test': DataLoader(i_test_subset, batch_size=batch_size, shuffle=False,
                              worker_init_fn=seed_worker, generator=g)
        },
        'cross': {
            'balanced_train_imbalanced_test': DataLoader(i_test_subset, batch_size=batch_size, shuffle=False,
                                                        worker_init_fn=seed_worker, generator=g),
            'imbalanced_train_balanced_test': DataLoader(b_test_subset, batch_size=batch_size, shuffle=False,
                                                        worker_init_fn=seed_worker, generator=g)
        }
    }
    
    return loaders, balanced_path, imbalanced_path


if __name__ == "__main__":
    # Example usage
    save_dir = '../src/Dataset/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create and visualize datasets
    balanced_dataset = PhoneNumberDataset(num_samples=1000, is_balanced=True, save_dir=save_dir)
    balanced_dataset.save_dataset("phone_balanced_sample.pt")
    balanced_dataset.visualize_samples(5)
    
    imbalanced_dataset = PhoneNumberDataset(num_samples=1000, is_balanced=False, save_dir=save_dir)
    imbalanced_dataset.save_dataset("phone_imbalanced_sample.pt")
    imbalanced_dataset.visualize_samples(5)
    
    # Get data loaders
    loaders, balanced_path, imbalanced_path = get_data_loaders(
        create_new=True, 
        num_samples=20000, 
        batch_size=64, 
        save_dir=save_dir
    )
    
    print(f"Balanced dataset path: {balanced_path}")
    print(f"Imbalanced dataset path: {imbalanced_path}")
    
    # Sanity check: verify a few samples
    train_iter = iter(loaders['balanced']['train'])
    images, labels = next(train_iter)
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"First few labels: {labels[0]}")