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

# ----------- Self-Attention -----------
# ----------- Self-Attention Block (Đã sửa đổi) -----------
# ----------- Self-Attention Block (Đã sửa đổi) -----------
class SelfAttention(nn.Module):
    """
    Block Self-Attention (Đã sửa lỗi CUDA assert).
    Mặc dù phiên bản cũ không sai về logic, nhưng đôi khi
    cách sắp xếp tensor có thể gây ra lỗi không mong muốn trên GPU.
    Phiên bản này sử dụng một cách viết phổ biến và ổn định hơn.
    """

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.key = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        # Chuyển đổi tensor để tính toán
        proj_query = self.query(x).reshape(B, -1, H * W)  # B x (C/8) x N
        proj_key = self.key(x).reshape(B, -1, H * W)  # B x (C/8) x N
        proj_value = self.value(x).reshape(B, -1, H * W)  # B x C x N

        # Tính attention map
        # (B x N x C/8) x (B x C/8 x N) -> (B x N x N)
        attention = torch.bmm(proj_key.permute(0, 2, 1), proj_query)
        attention = F.softmax(attention, dim=-1)

        # Áp dụng attention map lên value
        # (B x C x N) x (B x N x N) -> (B x C x N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # Reshape lại về kích thước ban đầu
        out = out.view(B, C, H, W)

        # Thêm kết quả vào đầu vào ban đầu
        return self.gamma * out + x


# ----------- Residual Block -----------
class ResidualBlock(nn.Module):
    """
    Block dư cơ bản để thêm vào giữa các lớp.
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


# ----------- SE Block (Channel Attention) -----------
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block để tạo attention theo kênh.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(in_channels // reduction, 1), in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


# ----------- Downsampling Block -----------
def down_block(in_channels, out_channels, use_attn=False):
    """
    Hàm tạo Downsampling Block với các lớp tiêu chuẩn và SEBlock.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        SEBlock(out_channels)
    ]
    if use_attn:
        layers.append(SelfAttention(out_channels))
    return nn.Sequential(*layers)


# ----------- Upsampling Block -----------
def up_block(in_channels, out_channels, dropout=False, use_attn=False):
    """
    Hàm tạo Upsampling Block với các lớp tiêu chuẩn và SEBlock.
    """
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        SEBlock(out_channels)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    if use_attn:
        layers.append(SelfAttention(out_channels))
    return nn.Sequential(*layers)


# ----------- Generator Main Model -----------
class UNetGenerator(nn.Module):
    """
    Mô hình Generator dạng U-Net cải tiến để chuyển ảnh xám sang màu.
    Đầu vào: 1 kênh (L*)
    Đầu ra: 2 kênh (a* và b*)
    """

    def __init__(self, input_channels=1, output_channels=2):
        super(UNetGenerator, self).__init__()
        # Downsampling
        self.d1 = down_block(input_channels, 64)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)

        # Middle
        self.middle = nn.Sequential(
            ResidualBlock(512),
            SelfAttention(512),
            ResidualBlock(512)
        )

        # Upsampling
        self.u1 = up_block(512 * 2, 256, dropout=True)
        self.u2 = up_block(256 * 2, 128, use_attn=True)
        self.u3 = up_block(128 * 2, 64)
        self.u4 = up_block(64 * 2, 32)

        # Final conv + thêm 1 residual block trước Tanh
        self.final = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, output_channels, 3, 1, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        m = self.middle(d4)

        u1 = self.u1(torch.cat([m, d4], dim=1))
        u2 = self.u2(torch.cat([u1, d3], dim=1))
        u3 = self.u3(torch.cat([u2, d2], dim=1))
        u4 = self.u4(torch.cat([u3, d1], dim=1))

        out = self.final(u4)

        return out


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
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load("E:/thuctap/output7/generator_best.pth", map_location=device))
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