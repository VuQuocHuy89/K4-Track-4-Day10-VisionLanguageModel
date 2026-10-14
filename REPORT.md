Họ tên: Vũ Quốc Huy
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,0843 | [0,0564; 0,1150] | 0,682 | 0,1727 | 0,0179 | 0,1557 |
| E1 | LoRA LLM | 33.030.144 | 5,69 | 37,89 | 0,2672 | [0,2311; 0,3081] | 1,170 | 0,2121 | 0,1835 | 0,5238 |
| E2 | + projector | 87.609.344 | 8,44 | 47,18 | 0,0733 | [0,0504; 0,0997] | 1,338 | 0,0630 | 0,0376 | 0,1918 |
| E3 | + LoRA ViT | 93.900.800 | 5,21 | 63,38 | 0,0101 | [0,0034; 0,0185] | 1,923 | 0,0054 | 0,0163 | 0,0000 |
| E4 | data_size = 0,1; LoRA LLM | 33.030.144 | 4,18 | 52,21 | 0,1473 | [0,1190; 0,1783] | 2,048 | 0,0900 | 0,0685 | 0,4203 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Theo GUIDE, LLM và embedding khoảng 4,0 tỷ tham số, khối thị giác (ViT) khoảng 415 triệu. Khi chạy thực tế, phần projector mở huấn luyện có 54.579.200 tham số (visual.merger: 13.640.192; visual.deepstack_merger_list: 40.939.008). E1 chỉ cập nhật LoRA trên 252 module LLM (33.030.144 tham số). E2 thêm projector (87.609.344 tham số). E3 thêm LoRA trên 96 module ViT, tổng 348 module LoRA và 93.900.800 tham số huấn luyện.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate` = 0,7942 (247/311 output hợp lệ). Hai lỗi thường gặp là output không parse được (64 ảnh; ví dụ `vlcsnap-00200` lặp khóa `bbox_2d` nhiều lần) và bỏ sót vật thể (FN = 548). Ví dụ output thô nguyên văn của `vlcsnap-00201` là `[]`, dù ảnh này có một crack và một manhole trong nhãn thật.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nhãn và prompt dùng tọa độ chuẩn hóa 0–1000. Tôi chấm lại cùng `e1.jsonl` trên 311 ảnh test bằng `score.py`: `coord=norm1000` cho F1 = 0,2672 (TP = 146), còn `coord=pixel` cho F1 = 0,0000 (TP = 0, FP = 510, FN = 583). Khi coi output 0–1000 là pixel, scorer nhân tọa độ x với 1000/640 và y với 1000/360; box bị phóng to, lệch khỏi nhãn thật nên không đạt IoU 0,5.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
`encode_example` gán `labels=-100` cho toàn bộ token prompt và chỉ giữ nhãn phần trả lời; kiểm tra trong notebook xác nhận phần nhãn giải mã thành JSON đáp án. Nếu tính loss trên prompt/token ảnh, phần lớn tín hiệu tối ưu sẽ dùng để học lặp lại đầu vào thay vì dự đoán box, làm lệch mục tiêu của bài toán.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trọng số 4-bit không phù hợp để cập nhật trực tiếp bằng AdamW như tham số float. Code giữ khối visual ngoài phần lượng tử hóa và chuyển riêng tham số projector được huấn luyện sang float32. Điều này giúp gradient và trạng thái tối ưu cùng dtype; khi thử dùng gradient FP16, `GradScaler` và AdamW đã báo lỗi dtype. Float32 là lựa chọn của lần chạy này để huấn luyện ổn định, không phải điều kiện bắt buộc cho mọi cách huấn luyện projector.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các frame gần nhau trong cùng phiên quay rất giống nhau. Chia ngẫu nhiên từng ảnh sẽ đưa các frame cùng phiên vào cả train và test, gây rò rỉ và làm F1 test cao giả. `prepare_data.py` gán cả phiên vào một split; `meta.json` ghi dataset_version d10-21e4070301. Train có 1.422 ảnh, 1.074 pothole, 1.750 crack, 691 manhole; val có 276 ảnh, 66 pothole, 470 crack, 103 manhole; test có 311 ảnh, 121 pothole, 299 crack, 163 manhole.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 đạt F1 0,2672 [0,2311; 0,3081] với 5,69 GB và 37,89 phút; tốt nhất trong ba setup. E2 đạt 0,0733 [0,0504; 0,0997] với 8,44 GB và 47,18 phút. E3 đạt 0,0101 [0,0034; 0,0185] với 5,21 GB và 63,38 phút. Các khoảng tin cậy không chồng nhau; trong lần chạy này, mở projector và LoRA ViT không đáng chi phí. VRAM đo của E3 thấp hơn E2 dù có nhiều tham số huấn luyện hơn; đây là số thực tế từ các phiên chạy, không suy ra E3 luôn dùng ít VRAM hơn.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Trên 311 ảnh test, CI95 của E1 rộng 0,0770 điểm F1 (0,2311–0,3081); E2 rộng 0,0493 và E3 rộng 0,0151. E1 vượt E2/E3 và E0 với các khoảng không chồng nhau, nhưng một test nhỏ chưa đủ để khẳng định mức chênh lệch sẽ giữ nguyên trên video hoặc điều kiện đường khác. So sánh hai setup có CI chồng nhau cần thêm dữ liệu hoặc bootstrap cặp theo ảnh.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Với setup tốt nhất E1, crack kém nhất: F1 = 0,1835, FP = 285, FN = 240; pothole = 0,2121, manhole = 0,5238. Crack là vệt mảnh, dài và có ranh giới khó xác định. Ảnh `report/figures/e1_5_vlcsnap-00203.png` cho thấy box crack thật trải rộng mặt đường (xanh), còn hai box dự đoán (đỏ) chỉ phủ các đoạn nhỏ và lệch vị trí; ảnh này có TP=0, FP=2, FN=1. Đây là một ví dụ lỗi định vị, không đủ để kết luận mọi lỗi crack có cùng nguyên nhân.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
E4 chỉ đổi `data_size` trên nền E1: 10% train (142/1.422 ảnh) so với 100% train (dùng lại E1), cùng 300 bước tối ưu và các cấu hình còn lại. F1 giảm từ 0,2672 [0,2311; 0,3081] xuống 0,1473 [0,1190; 0,1783]; hai CI95 không chồng nhau. Loss cuối của bản 10% là 0,0022 nhưng F1 test thấp hơn, phù hợp với dấu hiệu học quá sát tập con. Kết luận dữ liệu ít hơn làm giảm kết quả trong lần chạy này; chưa khái quát cho mọi lần chọn tập con hay seed.

## Nhìn lại

Lúc chạy E2 đã gặp lỗi dtype giữa gradient FP16 và AdamW, sau đó hết VRAM; phải đưa tham số/gradient và trạng thái tối ưu được huấn luyện về float32, giảm batch còn 1 và tích lũy 8 lần. Công cụ chấm đầu ra không phát hiện được các lỗi huấn luyện này.
