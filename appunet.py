import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
# ----- Thiết lập -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 256

# ----- Generator giống với code gốc -----
import torch
import torch.nn as nn
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

class UnetBasic1(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(UnetBasic1, self).__init__()

        # Encoder (ít tầng, ít kênh)
        self.enc1 = self.contract_block(input_channels, 16, 4, 2, 1)  # 128 -> 64
        self.enc2 = self.contract_block(16, 32, 4, 2, 1)              # 64 -> 32
        self.enc3 = self.contract_block(32, 64, 4, 2, 1)              # 32 -> 16

        # Bottleneck
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Decoder (ít tầng)
        self.dec3 = self.expand_block(128+64, 64, 4, 2, 1)   # 16 -> 32
        self.dec2 = self.expand_block(64+32, 32, 4, 2, 1)    # 32 -> 64
        self.dec1 = self.expand_block(32+16, 16, 4, 2, 1)    # 64 -> 128

        # Output
        self.final = nn.Sequential(
            nn.Conv2d(16, output_channels, 3, 1, 1),
            nn.Tanh()
        )

    def contract_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m = self.middle(e3)

        d3 = self.dec3(torch.cat([m, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.final(d1)


# ----- Hàm tiền xử lý ảnh -----
def preprocess_image(image):
    gray = image.convert("L")  # chuyển ảnh sang grayscale
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(gray).unsqueeze(0)  # shape: (1, 1, H, W)
    return tensor.to(device), gray

# ----- Hàm hậu xử lý -----
from skimage.color import lab2rgb

def postprocess(output_ab, input_l):
    """
    output_ab: tensor (1, 2, H, W)
    input_l: tensor (1, 1, H, W)
    """
    # Chuyển từ [-1, 1] → [-128, 127]
    ab = output_ab.squeeze(0).detach().cpu().numpy() * 128  # (2, H, W)
    l = input_l.squeeze(0).detach().cpu().numpy() * 50 + 50  # (1, H, W)

    lab = np.concatenate([l, ab], axis=0).transpose(1, 2, 0)  # (H, W, 3)

    rgb = lab2rgb(lab)  # float64, [0, 1]
    rgb = np.clip(rgb, 0, 1)

    return Image.fromarray((rgb * 255).astype(np.uint8))


# ----- Load mô hình đã huấn luyện -----
generator = UnetBasic1().to(device)
generator.load_state_dict(torch.load("E:/thuctap/output7+10final/model_best.pth", map_location=device))
generator.eval()

# ====== Giao diện Streamlit ======
st.set_page_config(page_title="Màu Sắc Hóa Ảnh", layout="centered")

st.markdown("<h1 style='text-align: center;'>Tô màu ảnh tự động</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Chọn ảnh đen trắng để màu hóa", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load ảnh và tiền xử lý
    input_image = Image.open(uploaded_file).convert("RGB")
    input_tensor, gray_img = preprocess_image(input_image)

    # Nếu là file mới khác file trước -> reset kết quả cũ
    if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
        st.session_state["gray_img"] = gray_img
        st.session_state["output_image"] = None
        st.session_state["has_run"] = False
        st.session_state["last_file"] = uploaded_file.name

    # --- Hiển thị preview LỚN khi chưa chạy ---
    # Chỉ hiển thị khi chưa chạy màu (has_run == False)
    if not st.session_state.get("has_run", False):
        st.image(st.session_state["gray_img"], caption="Ảnh gốc (đen trắng)", use_container_width=True)

    # Nút bấm để chạy màu hóa
    if st.button("▶️ Chạy Màu Sắc Hóa", key="run"):
        with st.spinner("⏳ Đang xử lý..."):
            with torch.no_grad():
                output_tensor = generator(input_tensor)
            output_image = postprocess(output_tensor, input_tensor)
        # Lưu kết quả và đánh dấu đã chạy
        st.session_state["output_image"] = output_image
        st.session_state["has_run"] = True

    # --- Nếu đã chạy thì hiển thị 2 ảnh song song (và ẩn preview lớn) ---
    if st.session_state.get("has_run", False) and st.session_state.get("output_image") is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state["gray_img"], caption="Ảnh đen trắng", use_container_width=True)
        with col2:
            st.image(st.session_state["output_image"], caption="Ảnh màu sắc hóa", use_container_width=True)

        st.success("✅ Đã hoàn thành tô màu ảnh.")

        # Nút tải ảnh màu về (khi bấm download sẽ rerun, nhưng has_run vẫn True nên preview lớn không xuất hiện)
        from io import BytesIO
        buf = BytesIO()
        st.session_state["output_image"].save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Tải ảnh màu về",
            data=byte_im,
            file_name="anh_mau.png",
            mime="image/png"
        )