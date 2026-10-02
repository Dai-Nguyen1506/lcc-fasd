import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_transforms(is_train=True):
    """
    Hàm tạo bộ biến đổi hình ảnh (Data Augmentation).
    - Tập Train: Áp dụng Augmentation (đã loại bỏ Affine/Rotation để tránh viền đen), đầu ra 224x224.
    - Tập Val/Test: Chỉ Resize về 224x224 và Normalize.
    """
    if is_train:
        return transforms.Compose([
            # Đưa về kích thước cơ sở lớn hơn một chút để có không gian crop
            transforms.Resize((256, 256)),
            
            # Zoom ngẫu nhiên và cắt chuẩn về 224x224 (Mô phỏng khoảng cách/vị trí khuôn mặt)
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            
            # Rung động ánh sáng/màu sắc (Biên độ dao động 30%, hue cực nhỏ 2% để giữ màu da)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            
            # Thêm mờ ngẫu nhiên (Mô phỏng camera mất nét)
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            
            # Tăng độ nét ngẫu nhiên với xác suất 50% (Mô phỏng cảm biến điện thoại)
            transforms.RandomAdjustSharpness(sharpness_factor=4.0, p=0.5),
            
            # Lật ngang
            transforms.RandomHorizontalFlip(p=0.5),
            
            transforms.ToTensor(),
            
            # Chuẩn hóa theo ImageNet (Bắt buộc khi dùng Pre-trained models như ResNet/ConvNeXt)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            # Đưa trực tiếp về 224x224, không áp dụng nhiễu
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir="datasets/LCC_FASD", batch_size=32):
    """
    Hàm khởi tạo DataLoader cho quá trình huấn luyện, đánh giá và kiểm thử.
    Sử dụng WeightedRandomSampler để cân bằng tự động số lượng ảnh Real và Spoof trong mỗi batch.
    """
    # 1. Khởi tạo Dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=get_transforms(is_train=True))
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=get_transforms(is_train=False))
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=get_transforms(is_train=False))

    # 2. Xây dựng WeightedRandomSampler cho tập Train
    targets = train_dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    
    # Tính trọng số nghịch đảo: Nhãn ít ảnh (Real) sẽ có trọng số bốc thăm cao hơn
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    
    # Khởi tạo bộ lấy mẫu (replacement=True để cho phép bốc lặp lại ảnh)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 3. Đóng gói DataLoader
    # num_workers=4 giúp CPU xử lý ảnh đa luồng song song với quá trình GPU/XPU huấn luyện
    loader_args = {'num_workers': 4, 'pin_memory': True, 'drop_last': False}
    
    # Lưu ý: Bắt buộc bỏ shuffle=True khi dùng sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, **loader_args)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_args)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader