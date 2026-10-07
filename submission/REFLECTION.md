# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Bản fine-tune đạt target 0.970, thắng prompt tối ưu +0.205, vậy mà phán quyết là FAILED.
Tôi đã mặc định rằng "điểm bài toán đích tăng mạnh" đồng nghĩa với thành công. Cổng hồi quy
bắt được việc điểm kiến thức phổ thông tụt từ 0.791 xuống 0.722 — thứ tôi sẽ không bao giờ
đo nếu lab không bắt. Ngạc nhiên thứ hai là `attn_only` hoà `correct` ở 0.970: tôi chờ nó
thua, và nó không thua.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở chỗ tôi đoán. Tôi nghĩ phần huấn luyện sẽ tốn nhất, nhưng thứ tốn nhất là
phải chạy lại cả pipeline: lần đầu tôi chạy trên Colab mà không để ý ô 3 mặc định
`EVAL_LIMIT = "8"`, nên toàn bộ kết quả chỉ tính trên 8 mẫu và `verify.py` báo FAIL
"full eval set used". Tôi cũng chỉ tải về `adapters/correct`, thiếu ba adapter đối chứng mà
NB5 cần, nên không thể chỉ đo lại mà phải chạy lại từ đầu (lần hai tôi chuyển sang Kaggle).
Một tham số mặc định tôi không đọc đã tốn hơn một lần chạy đầy đủ.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin rằng train loss giảm đều và thấp là dấu hiệu đủ để nói model tốt. Số của tôi bác
điều đó theo cả hai chiều: `attn_only` có loss thấp nhất (0.538) nhưng không thắng trên
target, còn `wrong_lr` có loss 1.570 nhưng target là 0.000 tuyệt đối chứ không phải "kém
hơn một ít". Tôi cũng từng tin fine-tune chỉ *thêm* năng lực; mức tụt regression −0.069
cho thấy nó có thể lấy đi thứ khác trong lúc thêm.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để đọc repo và liệt kê việc cần làm, đọc các file trong `results/`,
chạy `verify.py`, viết các ô chạy trên Kaggle, và dựng bản nháp report. Những chỗ nó sai
hoặc thiếu:

- Ban đầu nó hướng dẫn tôi chỉ zip `results` và `adapters/correct`; đến khi cần chạy lại
  NB5 mới lộ ra là thiếu ba adapter đối chứng.
- Nó đưa ra bảng "việc cần làm" mà không cảnh báo ô Colab mặc định `EVAL_LIMIT = "8"`;
  lỗi đó chỉ được phát hiện sau khi tôi đã chạy xong.
- Trên bộ kết quả 8 mẫu, nó nhận xét `attn_only` thấp hơn `correct` trên target (0.938 so
  với 0.969) là "đúng kiểu nghịch lý mục 4.1 hỏi". Với 50 mẫu thì hai run hoà nhau. Nó có
  ghi chú là 8 mẫu chưa kết luận được, nhưng hướng diễn giải ban đầu vẫn là sai.

Chỗ nó có ích nhất là việc tôi lười làm: đối chiếu 6 ca sai với nhãn trong
`eval_target.jsonl` và tìm ra cả 6 đều là cụm "Khi nào tiện".

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Trước khi đụng tới GPU: dựng tập eval và đóng băng nó, gồm cả một tập regression đủ lớn —
15 câu của lab khiến mức tụt 0.069 chỉ tương đương khoảng một câu trả lời, quá thô để ra
quyết định deploy. Sau đó đo base model với một prompt tử tế. Trong lab này prompt đã đưa
target từ 0.000 lên 0.765 mà không tốn phút train nào; nếu con số đó đủ cho khách hàng thì
tôi không fine-tune.
