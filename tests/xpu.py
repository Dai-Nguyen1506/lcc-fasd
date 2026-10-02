import torch

# Kiểm tra xem XPU có sẵn sàng không
if torch.xpu.is_available():
    device_count = torch.xpu.device_count()
    print(f"✅ Đã tìm thấy {device_count} thiết bị XPU.")
    for i in range(device_count):
        print(f" - Thiết bị {i}: {torch.xpu.get_device_name(i)}")
else:
    print("❌ XPU chưa sẵn sàng. Hãy kiểm tra lại cài đặt driver Intel Compute Runtime (Level Zero / OpenCL).")