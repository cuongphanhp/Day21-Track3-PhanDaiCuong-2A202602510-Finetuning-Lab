# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Minh Hiếu  **MSSV**: 2A202602669  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây lấy từ file trong `results/`. Tên file nguồn được ghi cạnh từng bảng.

---

## 1. Setup

**Lựa chọn và lý do.** Tôi giữ nguyên cấu hình mặc định của lab: tier `T4`, base model
`unsloth/Qwen3.5-4B`, corpus 250 ticket CSKH tiếng Việt → JSON triage 4 trường. Lý do: đây
là lần đầu tôi chạy pipeline này, và thời gian chạy trong `docs/MEASURED-T4-2026-08-20.md`
được đo trên đúng cấu hình đó, nên nếu số của tôi lệch thì tôi biết là do tôi chứ không do
model hay dữ liệu khác. Bài toán này cũng có thang đo khách quan (so khớp từng trường với
nhãn), không cần LLM judge. Tôi không sửa `OPTIMIZED_PROMPT` và không sửa tập eval.

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (seed 42, `train_frac=0.9`) |
| Eval | 50 ticket target + 15 câu regression, chạy đủ (`eval_limit: null`, `smoke_mode: false`) |
| `max_length` | 1024 (mặc định của tier T4) — p95 đo được là 98, max là 101 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch = 30 step (batch 1 × grad_accum 16) |
| LoRA `correct` | text-linear, 12 loại module, r=16, alpha=32, LR 1e-4, fp16 |

**Về `max_length`.** NB1 gợi ý 256 dựa trên p95 = 98; tier T4 đặt 1024. Tôi giữ 1024 vì
mẫu dài nhất chỉ 101 token, nên cả hai giá trị đều không cắt mẫu nào và không làm khác kết
quả huấn luyện. Nếu dữ liệu dài hơn, tôi sẽ hạ về giá trị theo p95 để tiết kiệm bộ nhớ.

**Template có giữ khối `<think>` không?** Có — `open_tag_present: true`,
`body_present: true`, verdict "reasoning preserved — safe to train on traces"
*(results/template_check.json)*. Tuy nhiên dữ liệu huấn luyện của tôi không có nội dung suy
luận: khối `<think>` trong mẫu là rỗng (xem mục 2), nên `valid_trace_rate` của bản
fine-tune là 0.0 (mục 5).

---

## 2. Mask proof (NB1)

*(results/mask_proof.json)*

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39 / 94 token) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Đoạn được tính loss (giải mã ngược từ các token có nhãn):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (không tính loss) kết thúc ở `<|im_start|>assistant\n<think>\n\n`: toàn bộ
system prompt, ticket của người dùng và thẻ mở `<think>` đều nằm ngoài loss. Tỉ lệ 41% nằm
xa ngưỡng 95% — model chỉ bị phạt trên câu trả lời JSON, không học chép lại câu hỏi.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

*(results/baselines_frozen.json cho (a), (b); results/verdict.json cho (c); n = 50)*

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.00 | 3218.8 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.00 | 1018.4 |
| (c) LoRA fine-tune | 0.970 | 0.7222 | 1.00 | 1409.1 |

**(b) có thật sự mạnh hơn (a) không?** Có: target 0.000 → 0.765, format 0.00 → 1.00.
Với prompt ngây thơ ("Phân loại ticket sau."), base model không xuất ra JSON hợp lệ lần
nào, nên điểm target bằng 0 là do sai định dạng chứ chưa chắc do không hiểu ticket. Prompt
tối ưu mô tả schema, liệt kê giá trị hợp lệ và có ví dụ, đủ để format đạt 100%.

**Tôi có sửa `OPTIMIZED_PROMPT` không?** Không. SHA ghi trong file đóng băng
(`719e74d3b6232053`) khớp với prompt gốc của lab; `verify.py` xác nhận mục này.

Điểm regression của (a) và (b) bằng nhau (0.7911) vì hai baseline dùng chung base model
và tập regression không dùng prompt triage.

---

## 4. Giải phẫu cấu hình sai (NB4)

*(results/runs.csv cho cấu hình, loss, thời gian, VRAM; results/autopsy.json cho target)*

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6258 | **0.970** | 426.0 | 8.78 |
| `attn_only` | q,v (2 module) | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5378 | **0.970** | 271.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 399.5 | 8.78 |
| `qlora` | text-linear, 4-bit | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 490.3 | 3.86 |

Cả bốn run chạy đúng 30 step. Mỗi run đối chứng chỉ đổi **một** biến so với `correct`:

- `attn_only`: đổi **vị trí gắn adapter** (q,v thay cho mọi lớp linear của phần text).
  Rank được nâng lên 283 để số tham số huấn luyện khớp — lệch 8,192 tham số, tức 0.03%.
- `wrong_lr`: đổi **learning rate** (1e-5 thay cho 1e-4).
- `qlora`: đổi **độ chính xác khi nạp base model** (4-bit thay cho 16-bit).

Xếp hạng theo target: `correct` = `attn_only` (0.970) > `qlora` (0.940) > `wrong_lr` (0.000).
Xếp hạng theo train loss: `attn_only` (0.538) < `correct` (0.626) < `qlora` (0.706) <
`wrong_lr` (1.570). Hai thứ tự **không giống nhau** ở cặp đứng đầu.

**4.1 — `attn_only` so với `correct`: vị trí hay rank?**

Trên tập target, `attn_only` **hoà** `correct`: cả hai đạt 0.970, tức cùng sai 6 trên 200
trường. Thứ tự theo train loss thì khác: `attn_only` có loss thấp hơn rõ (0.538 so với
0.626), nên nếu chỉ nhìn loss tôi sẽ kết luận `attn_only` tốt hơn — và kết luận đó không
được tập target ủng hộ. Với cùng ngân sách ~32,46 triệu tham số, số đo của tôi **không**
cho thấy vị trí gắn adapter là đòn bẩy trên bài toán này: gắn vào q,v với rank 283 cho kết
quả ngang với gắn vào 12 loại module với rank 16. Tôi không coi đây là bằng chứng bác bỏ
kết luận "vị trí quan trọng hơn rank" của bài giảng, vì bài toán của tôi đã gần chạm trần
(0.970) và tập eval chỉ có 50 mẫu — khi cả hai cấu hình cùng giải gần hết bài toán thì
phép đo không còn đủ độ phân giải để tách chúng. Điều tôi kết luận được là: với một bài
toán phân loại hẹp, dữ liệu sinh theo khuôn như thế này, cả hai vị trí đều đủ. Khác biệt
tôi đo được nằm ở chi phí: `attn_only` train nhanh hơn 36% (271.9 s so với 426.0 s) và
suy luận nhanh hơn (923.7 ms so với 1409.1 ms mỗi mẫu).

**4.2 — `wrong_lr`: chỉ khác một con số**

`wrong_lr` dùng LR 1e-5, nhỏ hơn `correct` đúng 10 lần, mọi thứ khác giữ nguyên. Sau cùng
30 step, loss của nó dừng ở 1.570 trong khi `correct` xuống 0.626. Trên tập target hậu quả
là tuyệt đối: target 0.000 và format 0.00, tức adapter chưa dịch chuyển model đủ để xuất
JSON hợp lệ dù chỉ một lần — nó hành xử như baseline (a). Nếu chỉ nhìn đường loss mà không
biết LR, tôi dễ kết luận sai rằng "dữ liệu quá ít" hoặc "cần train thêm nhiều epoch", rồi
tốn thêm giờ GPU, trong khi nguyên nhân thật là thang LR của full fine-tune bị áp cho
LoRA. Run này cũng cho thấy loss 1.57 không có nghĩa là "học được một nửa": với bài toán
đòi đúng định dạng, kết quả là 0.

**4.3 — `qlora`: tiết kiệm gì, trả giá gì**

`qlora` giảm VRAM đỉnh từ 8.78 GB xuống 3.86 GB, tức tiết kiệm 4.92 GB (56%). Cái giá đo
được gồm ba phần: target giảm từ 0.970 xuống 0.940 (thêm 6 trường sai trên 200), thời gian
train tăng 15% (490.3 s so với 426.0 s), và suy luận chậm hơn (1797.4 ms so với 1409.1 ms
mỗi mẫu). Số đo của tôi ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" **trong
điều kiện có đủ VRAM**: trên T4 16 GB, bản 16-bit chạy vừa với 8.78 GB, nên không có lý do
gì để nhận mức giảm chất lượng và tốc độ đó. Mức giảm 0.030 trên 50 mẫu là nhỏ và tôi chưa
chạy nhiều seed để biết nó có ổn định không; nếu chỉ có GPU 6–8 GB thì QLoRA vẫn là lựa
chọn chạy được, với target 0.940 vẫn cao hơn nhiều so với baseline (b) 0.765.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.069` · `valid_trace_rate = 0.00`
*(results/verdict.json)*

Lý do ghi trong file: *"general capability regressed by 0.069 (tolerance 0.020)"*.

**Diễn giải.** Bản fine-tune thắng rõ trên bài toán đích: target tăng từ 0.765 lên 0.970
so với base model đã được prompt tử tế, format giữ 100%. Nhưng cổng hồi quy có hai điều
kiện, và nó trượt điều kiện thứ hai: điểm regression (đo bằng tỉ lệ từ khoá đúng trên 15
câu hỏi kiến thức phổ thông) tụt từ 0.7911 xuống 0.7222, vượt ngưỡng cho phép 0.020 hơn ba
lần. Theo đúng luật đã đặt ra trước khi train, bản này **không được deploy** như một model
đa dụng.

Tôi đọc kết quả này với hai lưu ý. Thứ nhất, tập regression chỉ có 15 câu, nên mức tụt
0.069 tương đương khoảng một câu trả lời (0.069 × 15 ≈ 1.03 điểm). Một phép đo thô như vậy
không cho tôi biết mức tụt thật là 2% hay 12%. Nhưng lưu ý này không cho phép tôi nới
cổng: ngưỡng được đặt trước, và cách đúng để phản biện nó là đo trên tập regression lớn
hơn, không phải bỏ qua kết quả. Thứ hai, hướng của tín hiệu là hợp lý về mặt cơ chế: dữ
liệu train 100% là ticket → JSON, không có mẫu phổ thông nào, và adapter gắn vào mọi lớp
linear của phần text, nên việc model lệch khỏi hành vi trả lời tự do là điều có thể xảy
ra. Tôi chưa xem từng câu regression bị sai nên chưa xác định được nó hỏng theo kiểu nào.

`valid_trace_rate = 0.00` không phải là sụp đổ suy luận do fine-tune gây ra theo nghĩa
đáng lo: dữ liệu train của tôi vốn có khối `<think>` rỗng, nên model được dạy đúng là bỏ
qua suy luận. Với bài toán phân loại ra JSON thì đó là hành vi mong muốn, nhưng nó cũng
có nghĩa adapter này không dùng được cho việc cần suy luận từng bước.

Điều kết quả này nói về bài toán của tôi: fine-tune giải được phần mà prompt không giải
nổi (+0.205), nhưng với cấu hình hiện tại nó chỉ phù hợp làm một adapter **chuyên dụng**,
bật riêng cho luồng triage ticket, chứ không thay thế base model.

---

## 6. Định tính — có cả ca THUA

*(results/qualitative.json cho dự đoán của fine-tune; nhãn đúng lấy từ
data/eval_target.jsonl theo chỉ số `i`)*

`qualitative.json` chỉ lưu dự đoán của bản fine-tune, không lưu câu trả lời từng mẫu của
baseline (b), nên tôi không lập được cột "(b) prompt" và không khẳng định được ca nào là
"fine-tune thắng (b)" ở mức từng mẫu. Bảng dưới so bản fine-tune với nhãn đúng.

| # | i | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | 3 | "…bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | hoan_tien · **trung_binh** · bình giữ nhiệt · … | ❌ **FT sai** urgency (0.75) |
| 2 | 5 | "…nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi." | san_pham_loi · **thap** · nồi chiên không dầu · trung_tinh | san_pham_loi · **trung_binh** · nồi chiên không dầu · … | ❌ **FT sai** urgency (0.75) |
| 3 | 39 | "…nồi chiên không dầu mã đơn VN949966. Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · nồi chiên không dầu · tieu_cuc | hoan_tien · **trung_binh** · nồi chiên không dầu · … | ❌ **FT sai** urgency (0.75) |
| 4 | 27 | "…bàn phím cơ mã đơn VN130559. Cho tôi trả lại. Ngay lập tức. Mình vẫn tin tưởng shop." | doi_tra · cao · bàn phím cơ · tich_cuc | doi_tra · cao · bàn phím cơ · tich_cuc | ✅ FT đúng 4/4 |
| 5 | 30 | "…đèn bàn LED mã đơn OD936122. Hoàn lại. Sớm nhé. Lần cuối mua ở đây." | doi_tra · trung_binh · đèn bàn LED · tieu_cuc | doi_tra · trung_binh · đèn bàn LED · … | ✅ FT đúng 4/4 — "Hoàn lại" được gán đúng là `doi_tra`, không nhầm sang `hoan_tien` |
| 6 | 2 | "…đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều." | hoan_tien · cao · đèn bàn LED · tich_cuc | hoan_tien · cao · đèn bàn LED · … | ✅ FT đúng 4/4 |

(Dấu "…" ở cột fine-tune là chỗ `qualitative.json` cắt chuỗi dự đoán ở 90 ký tự; điểm
0.75 hay 1.0 của từng mẫu cho biết các trường còn lại đúng.)

**Mẫu chung ở các ca FT sai.** Có, và rất rõ. Bản fine-tune sai tổng cộng 6 mẫu
(i = 3, 5, 12, 39, 41, 46), mỗi mẫu sai đúng một trường, và cả 6 đều là **cùng một lỗi**:
ticket chứa cụm "Khi nào tiện", nhãn đúng là `urgency: thap`, model đoán `trung_binh`.
Đây cũng là toàn bộ 6 ticket có cụm này trong tập eval — model sai 6/6. Hai cụm khác cùng
mang nghĩa `thap` thì model đúng hết: "Không vội" 7/7 và "Hỏi cho biết thôi" 5/5.

Điều đáng chú ý là lỗi này không do thiếu dữ liệu: tập train có 35 ticket chứa "Khi nào
tiện" và cả 35 đều gán `thap`. Tôi chưa xác định được vì sao model không học được ánh xạ
này sau 30 step. Một giả thuyết là "Khi nào tiện" đọc lên giống một câu hỏi về thời gian
nên model nghiêng về mức trung bình, trong khi "Không vội" phủ định sự gấp một cách tường
minh; tôi chưa kiểm chứng giả thuyết đó. Toàn bộ 6 trường sai của bản fine-tune (0.970 =
194/200) quy về đúng một cụm từ này.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này để thay base model, và tôi sẽ cân
nhắc deploy nó như một adapter chuyên dụng cho riêng luồng triage ticket. Lý do của vế
đầu là cổng hồi quy: điểm regression tụt 0.069 trong khi ngưỡng đặt trước là 0.020, và
tôi không có căn cứ để bỏ qua một quy tắc mình đã chấp nhận trước khi thấy kết quả. Lý do
của vế sau là trên đúng bài toán đích, fine-tune làm được điều mà prompt không làm được:
target tăng từ 0.765 lên 0.970, và toàn bộ lỗi còn lại quy về một cụm từ duy nhất ("Khi
nào tiện"), tức là một lỗi có thể khoanh vùng và sửa bằng dữ liệu.

Đòn bẩy thật sự trong lab này, theo số đo của tôi, là **learning rate**. Đổi LR từ 1e-4
xuống 1e-5 đưa target từ 0.970 về 0.000 — không biến nào khác tạo ra khoảng cách cỡ đó.
Độ chính xác khi nạp model (QLoRA) đứng thứ hai với mức giảm 0.030. Vị trí gắn adapter,
thứ tôi tưởng sẽ quan trọng nhất, lại không tạo ra khác biệt đo được: `attn_only` hoà
`correct` ở 0.970 khi ngân sách tham số được khớp. Tôi nghĩ nguyên nhân là bài toán quá dễ
so với sức chứa của adapter nên cả hai cấu hình cùng chạm trần, chứ không phải vị trí vô
nghĩa nói chung. Mask là điều kiện nền: nếu mask sai thì mọi so sánh ở trên đều vô giá
trị, nhưng khi đã đúng thì nó không còn là biến để tối ưu.

Bài học lớn nhất về phương pháp là train loss không xếp hạng được các run: `attn_only` có
loss thấp nhất nhưng không thắng trên target, và nếu không có cổng hồi quy thì tôi đã báo
cáo bản fine-tune là một thành công trọn vẹn.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Kết luận đổi chiều khi tăng cỡ mẫu.** Lần chạy đầu của tôi vô tình ở chế độ rút gọn
   `EVAL_LIMIT=8` (giá trị mặc định trong ô Colab), và phán quyết là PASSED với
   regression Δ = +0.000. Chạy lại đủ 50 ticket + 15 câu regression thì cùng một cấu hình
   cho ra FAILED với regression Δ = −0.069. Nếu tôi viết report từ lần chạy đầu, tôi đã
   báo cáo ngược với sự thật. Từ giờ tôi kiểm tra `n` trước khi đọc bất kỳ điểm số nào.
2. **Điểm tổng 0.970 che mất một lỗi có hệ thống.** Tôi tưởng 3% sai là nhiễu rải rác.
   Khi mở từng ca thì cả 6 trường sai đều là một lỗi: cụm "Khi nào tiện" bị gán
   `trung_binh` thay vì `thap`, sai 6/6, trong khi tập train có 35 mẫu dạy đúng điều đó.
   Một con số trung bình không cho tôi biết model sai *đều* hay sai *trọn một nhóm*; chỉ
   đọc từng ca mới thấy, và với khách hàng thật thì "sai trọn một kiểu ticket" nghiêm
   trọng hơn nhiều so với "sai 3%".
3. **Train loss thấp hơn không có nghĩa là model tốt hơn.** `attn_only` kết thúc với
   loss 0.538, thấp hơn `correct` (0.626), nhưng hai run hoà nhau ở target 0.970. Ngược
   lại, `wrong_lr` có loss 1.570 — nhìn như "học chậm" — nhưng target là 0.000 chứ không
   phải "kém hơn một chút". Loss là thứ tôi nhìn để biết quá trình train có chạy hay
   không, không phải thứ để chọn model.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu phổ thông vào tập train (replay)
rồi chạy lại NB3 + NB5, để xem regression có quay về trong ngưỡng 0.020 mà vẫn giữ được
target hay không; và xem từng câu regression bị sai để biết model hỏng theo kiểu nào.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

Tôi không làm phần thưởng nào.
