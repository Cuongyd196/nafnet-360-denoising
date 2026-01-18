# Ghi chú về NAFNet

- NAFNet được sử dụng làm mạng nền (backbone) và không thay đổi kiến trúc gốc.

- Các mặt Cubemap được chuẩn hóa về dải giá trị [0,1] nhằm đảm bảo ổn định số học trong quá trình suy luận.

- Các vùng gần như phẳng (ví dụ: bầu trời hoặc bề mặt đồng nhất) được bỏ qua khi khử nhiễu để tránh các vấn đề số học.

- Pipeline đề xuất phù hợp cho các ứng dụng xử lý ảnh toàn cảnh, thực tế ảo (VR) và tham quan ảo (VR tour).