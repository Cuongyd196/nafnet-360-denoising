# Khử nhiễu ảnh toàn cảnh 360° dựa trên Cubemap và NAFNet

Dự án này đề xuất một **pipeline khử nhiễu ảnh toàn cảnh (360°)** dựa trên
**biểu diễn Cubemap** và **mạng học sâu NAFNet**.

Thay vì xử lý trực tiếp ảnh equirectangular (ERP) vốn chứa nhiều biến dạng hình học,
ảnh đầu vào được chuyển đổi thành **sáu mặt Cubemap**, khử nhiễu độc lập trên từng mặt,
sau đó ghép lại và chuyển ngược về dạng ERP.

---

## Pipeline xử lý

1. Chuyển ảnh ERP → Cubemap (6 mặt)
2. Khử nhiễu từng mặt bằng NAFNet
3. Ghép Cubemap theo bố cục Dice
4. Chuyển Cubemap → ERP
5. (Tùy chọn) Trộn với ảnh gốc để giữ chi tiết

---

## Cấu trúc thư mục

```text
nafnet-360-denoising/
├─ basicsr/
│  └─ denoise_360_cubemap.py
├─ demo/
│  ├─ cubemap_faces_raw/
│  ├─ cubemap_faces_denoised/
│  └─ test208.jpg
├─ options/
│  └─ test/SIDD/NAFNet-width64.yml
├─ pretrained_models/
│  └─ NAFNet-SIDD-width64.pth
```

## Cài đặt, mô hình và cách sử dụng
### Yêu cầu cài đặt NAFNet

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/cuongyd196/nafnet-360-denoising
cd nafnet-360-denoising
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```
### Cài đặt thư viện bổ sung để xử lý ảnh toàn cảnh
```bash
pip install imageio py360convert
```

### Tải mô hình NAFNet đã huấn luyện trước
Tải mô hình NAFNet - NAFNet-SIDD-width64  đã huấn luyện trên tập SIDD từ [đây](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) và đặt vào thư mục `experiments/pretrained_models/`.

### Cách sử dụng
Chạy script khử nhiễu ảnh toàn cảnh với lệnh sau:

```bash
python basicsr/denoise_360_cubemap.py \
  -opt options/test/SIDD/NAFNet-width64.yml \
  --input_path ./demo/test208.jpg \
  --output_path ./demo/denoise_test208.jpg
```

### Kết quả đầu ra

Kết quả khử nhiễu sẽ được lưu tại `./demo`.
    
- Ảnh toàn cảnh (ERP) sau khử nhiễu

- Ảnh ERP sau khi trộn với ảnh gốc nhằm giữ chi tiết tốt hơn
- Ảnh các mặt Cubemap thô và đã khử nhiễu sẽ được lưu trong các thư mục `cubemap_faces_raw/` và `cubemap_faces_denoised/` tương ứng.
---

# Ghi chú 
- NAFNet được sử dụng làm mạng nền (backbone) và không thay đổi kiến trúc gốc.

# Tham khảo

- [NAFNet](https://github.com/megvii-research/NAFNet)