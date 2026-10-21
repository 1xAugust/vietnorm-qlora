# Lộ trình

File này là nguồn duy nhất cho tiến độ và các hạng mục nghiên cứu dự kiến của dự án.

- [x] Tạo cấu trúc public starter tối giản và README
- [x] Bổ sung identity baseline chạy trên CPU và unit test cho các metric cơ bản
- [x] Bổ sung script chuẩn bị split `train`/`dev`/`test` đã lọc của ViLexNorm
- [ ] Xác minh QLoRA training end-to-end trên Colab T4 và cố định dependency versions
- [ ] Tái lập untuned Qwen baseline trên held-out test
- [ ] Training LoRA adapters với các nested subset 1k/3k/5k
- [ ] Triển khai ERR token-aligned tương thích bài báo kèm test
- [ ] Xây dựng semantic-preservation test set và unchanged-sentence test
- [ ] Bổ sung kết quả thật có thể tái lập, tagged release đầu tiên, Colab notebook và demo
- [ ] Ghi nhận hoạt động bảo trì và review Pull Request từ cộng đồng
