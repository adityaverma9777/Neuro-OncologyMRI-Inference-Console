import os

import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO

YOLO_WEIGHTS = "best.pt"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

WINDOWS_XP_COLORS = {
    "bg": "#ece9d8",
    "title": "#0053e1",
    "status": "#f3f3f3",
    "border": "#808080",
}

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

custom_css = f"""
body {{ background: {WINDOWS_XP_COLORS["bg"]}; font-family: Tahoma, Verdana, sans-serif; }}
.gradio-container {{
    border: 2px solid {WINDOWS_XP_COLORS["border"]};
    background: {WINDOWS_XP_COLORS["bg"]};
    border-radius: 6px;
    max-width: 700px;
    margin: 32px auto;
    box-shadow: 0 4px 16px #bbb;
}}
.gradio-title {{
    background: {WINDOWS_XP_COLORS["title"]};
    color: #fff;
    padding: 10px 16px;
    font-size: 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-bottom: 0;
}}
.status-bar {{
    background: {WINDOWS_XP_COLORS["status"]};
    color: #333;
    padding: 6px 16px;
    font-size: 13px;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
    border-top: 1px solid {WINDOWS_XP_COLORS["border"]};
    margin-top: 0;
}}
"""


class DetectionModule:
    def __init__(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        self.model = YOLO(weights_path)

    def run(self, image):
        if image is None:
            return []
        results = self.model(image, verbose=False)
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class": names.get(cls_idx, str(cls_idx)),
                        "conf": conf,
                        "box": [x1, y1, x2, y2],
                    }
                )
        return detections


class ExplanationModule:
    def __init__(self, api_key, api_url=GROQ_API_URL):
        self.api_key = api_key
        self.api_url = api_url

    def generate(self, detections):
        if not self.api_key:
            return "[Groq API key not set. Cannot generate explanation.]"
        if not detections:
            return "No tumor detected with sufficient confidence."
        det_lines = [f"- Tumor type: {d['class']}, Confidence: {d['conf']:.2f}" for d in detections]
        prompt = (
            "You are a medical AI assistant.\n"
            "Input:\n"
            f"Detection count: {len(detections)}\n"
            + "\n".join(det_lines)
            + "\nExplain in simple terms:\n"
            "- What was detected\n"
            "- What confidence means\n"
            "- Avoid medical diagnosis\n"
            "- Add disclaimer\n"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.2,
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"[Groq API error: {exc}]"


class VisualizationPipeline:
    def __init__(self):
        self.font = ImageFont.load_default()
        self.box_color = (0, 83, 225)
        self.text_color = (0, 0, 0)

    def draw(self, image, detections):
        rendered = image.convert("RGB").copy()
        draw = ImageDraw.Draw(rendered)
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["box"])
            label = f"{detection['class']} ({detection['conf']:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline=self.box_color, width=3)
            draw.text((x1, max(0, y1 - 16)), label, fill=self.text_color, font=self.font)
        return rendered


class InferenceOrchestrator:
    def __init__(self, detection_module, explanation_module, visualization):
        self.detection = detection_module
        self.explanation = explanation_module
        self.visualization = visualization

    def predict(self, image):
        detections = self.detection.run(image)
        visual = self.visualization.draw(image, detections)
        explanation = self.explanation.generate(detections)
        if detections:
            top = max(detections, key=lambda item: item["conf"])
            return visual, top["class"], top["conf"], explanation
        return visual, "no tumor", 0.0, explanation


detection_module = DetectionModule(YOLO_WEIGHTS)
explanation_module = ExplanationModule(GROQ_API_KEY)
visualization = VisualizationPipeline()
orchestrator = InferenceOrchestrator(detection_module, explanation_module, visualization)


def set_ready():
    return "Ready"


def analyze(image):
    if image is None:
        return "Upload an MRI image to analyze.", None, "", 0.0, ""
    visual, tumor, conf, expl = orchestrator.predict(image)
    return "Analysis complete.", visual, tumor, conf, expl


with gr.Blocks(title="Neuro-Oncology MRI Inference Console") as demo:
    gr.Markdown(
        "<div class='gradio-title'>Neuro-Oncology MRI Inference Console</div>"
        "<div class='status-bar'>YOLO-based lesion localization with structured LLM-assisted explanation for research workflows.</div>"
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Upload MRI Image", elem_id="img-in")
            status = gr.Markdown("Initializing inference pipeline...", elem_id="status-bar")
        with gr.Column():
            image_out = gr.Image(type="pil", label="Annotated MRI Output", elem_id="img-out")
            tumor_type = gr.Textbox(label="Predicted Finding", interactive=False)
            confidence = gr.Number(label="Detection Confidence", interactive=False)
            explanation = gr.Textbox(label="Structured Interpretation Summary", lines=6, interactive=False)
    demo.load(set_ready, None, status)
    analyze_btn = gr.Button("Run Inference", elem_id="analyze-btn", interactive=True)
    analyze_btn.click(
        analyze,
        inputs=[image_in],
        outputs=[status, image_out, tumor_type, confidence, explanation],
    )
    gr.Markdown("<div class='status-bar'>For research use only. Not for clinical diagnosis.</div>")


if __name__ == "__main__":
    launch_kwargs = {
        "theme": gr.themes.Base(),
        "css": custom_css,
        "show_error": True,
    }
    if os.getenv("SPACE_ID"):
        launch_kwargs["server_name"] = "0.0.0.0"
        port = os.getenv("PORT")
        if port:
            launch_kwargs["server_port"] = int(port)
    demo.launch(**launch_kwargs)
