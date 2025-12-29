class_names_5 = [
'Neutrality in learning state.',
'Enjoyment in learning state.',
'Confusion in learning state.',
'Fatigue in learning state.',
'Distraction.'
]

class_names_with_context_5 = [
'an expression of Neutrality in learning state.',
'an expression of Enjoyment in learning state.',
'an expression of Confusion in learning state.',
'an expression of Fatigue in learning state.',
'an expression of Distraction.'
]


##### onlyface
class_descriptor_5_only_face = [
'Relaxed mouth,open eyes,neutral eyebrows,smooth forehead,natural head position.',

'Upturned mouth,sparkling or slightly squinted eyes,raised eyebrows,relaxed forehead.',

'Furrowed eyebrows, slightly open mouth, squinting or narrowed eyes, tensed forehead.',

'Mouth opens in a yawn, eyelids droop, head tilts forward.',

'Averted gaze or looking away, restless or fidgety posture, shoulders shift restlessly.'
]

##### with_context
class_descriptor_5 = [
'Relaxed mouth,open eyes,neutral eyebrows,no noticeable emotional changes,engaged with study materials, or natural body posture.',

'Upturned mouth corners,sparkling eyes,relaxed eyebrows,focused on course content,or occasionally nodding in agreement.',

'Furrowed eyebrows, slightly open mouth, wandering or puzzled gaze, chin rests on the palm,or eyes lock on learning material.',

'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, or hand writing.',

'Shifting eyes, restless or fidgety posture, relaxed but unfocused expression,frequently checking phone,or averted gaze from study materials.'
]

# ======================================================================
# Helper: trả về (class_names, input_text) dùng cho mô hình
# ======================================================================

def get_class_info(args):
    """
    Trả về:
        class_names: danh sách tên class ngắn gọn (dùng để hiển thị / vẽ confusion matrix)
        input_text : danh sách prompt đưa vào PromptLearner / TextEncoder

    args.text_type có 3 lựa chọn:
        - 'class_names'
        - 'class_names_with_context'
        - 'class_descriptor'
    """

    dataset = getattr(args, "dataset", "RAER")

    if dataset.upper() != "RAER":
        # Nếu sau này bạn có thêm dataset khác thì xử lý thêm ở đây
        raise ValueError(f"get_class_info hiện mới hỗ trợ dataset RAER, nhưng nhận '{dataset}'")

    # Mặc định tất cả đều dùng 5 class của RAER
    # class_names_5, class_names_with_context_5, class_descriptor_5
    text_type = getattr(args, "text_type", "class_descriptor")

    if text_type == "class_names":
        # input_text = chính tên class (rất ngắn, kiểu "Neutrality in learning state.")
        class_names = class_names_5
        input_text = class_names_5

    elif text_type == "class_names_with_context":
        # input_text = câu 'an expression of ...' (có chút ngữ cảnh)
        class_names = class_names_5
        input_text = class_names_with_context_5

    elif text_type == "class_descriptor":
        # ✅ cái bạn đang dùng: miêu tả chi tiết nét mặt / ánh mắt / tư thế, v.v.
        class_names = class_names_5
        input_text = class_descriptor_5

    else:
        raise ValueError(f"Unknown text_type: {text_type}")

    return class_names, input_text