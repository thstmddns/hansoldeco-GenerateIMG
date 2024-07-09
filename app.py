#  객체검출 -> 삭제 체크박스 적용본
import torch
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
from diffusers import StableDiffusionInpaintPipeline
import gradio as gr

# 모델 로드
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32)
pipe = pipe.to("cuda")

def detect_objects(image):
    # 객체 검출
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # 결과 후처리
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # 검출된 객체 정보 추출
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.9:
            box = [round(i) for i in box.tolist()]
            detected_objects.append({"label": model.config.id2label[label.item()], "box": box})
    
    return detected_objects

def display_detected_objects(image):
    detected_objects = detect_objects(image)
    labeled_image = image.copy()
    draw = ImageDraw.Draw(labeled_image)
    object_labels = []
    for obj in detected_objects:
        box = obj["box"]
        label = obj["label"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), label, fill="red")
        object_labels.append(f"{label} at {box}")
    return labeled_image, gr.update(choices=object_labels)

def inpaint_image(image, selected_objects):
    detected_objects = detect_objects(image)

    # 마스크 생성
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for obj in detected_objects:
        object_label = f"{obj['label']} at {obj['box']}"
        if object_label in selected_objects:
            box = obj["box"]
            draw.rectangle(box, fill=255)
    
    # Inpainting 수행
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    output = pipe(prompt="a modern interior", image=image, mask_image=mask).images[0]
    # output = pipe(prompt="remove", image=image, mask_image=mask).images[0]

    
    return output

# Gradio 인터페이스 설정
with gr.Blocks() as interface:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")
        labeled_image_output = gr.Image(label="Labeled Image")
    detect_button = gr.Button("Detect Objects")
    
    objects_list = gr.CheckboxGroup(label="Detected Objects")
    final_output = gr.Image(label="Output Image")
    
    inpaint_button = gr.Button("Remove Selected Objects")
    
    detect_button.click(fn=display_detected_objects, inputs=image_input, outputs=[labeled_image_output, objects_list])
    inpaint_button.click(fn=inpaint_image, inputs=[image_input, objects_list], outputs=final_output)

# Gradio 인터페이스 실행
interface.launch()
