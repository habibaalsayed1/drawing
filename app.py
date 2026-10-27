
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(
    page_title="AI Drawing Challenge",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #101827 0%, #172554 100%); color: white; }
    h1, h2, h3, p, label { color: white !important; }
    .hero { text-align:center; padding: 0.6rem 0 0.2rem 0; }
    .hero h1 { font-size: clamp(2rem, 5vw, 3.6rem); margin-bottom: 0.15rem; }
    .hero p { font-size: 1.15rem; color: #cbd5e1 !important; }
    .guess-card { background: #1e293b; border: 1px solid #334155; padding: 1rem; border-radius: 14px; margin: .4rem 0; }
    .small-note { color: #cbd5e1; font-size: .9rem; }
    div.stButton > button { border-radius: 12px; font-weight: 700; min-height: 3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎨 AI DRAWING CHALLENGE</h1>
  <p>You draw it. AI guesses it. Can you fool the machine?</p>
</div>
""", unsafe_allow_html=True)


MODEL_NAME = "kmewhort/beit-sketch-classifier"
def load_classifier():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return processor, model, device


def preprocess_canvas(rgba: np.ndarray) -> Image.Image:
    """Convert the canvas into a clean, centered sketch image."""

    if rgba is None or rgba.size == 0:
        raise ValueError("Draw something on the canvas first.")

    rgb = Image.fromarray(np.uint8(rgba)).convert("RGB")
    gray = ImageOps.grayscale(rgb)
    arr = np.asarray(gray)

    # Find the dark drawing strokes.
    ys, xs = np.where(arr < 235)

    if len(xs) == 0:
        raise ValueError("Draw something on the canvas first.")

    # Crop around the drawing.
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    crop = gray.crop((x_min, y_min, x_max, y_max))

    # Preserve the proportions and center the drawing.
    side = max(crop.width, crop.height) + 40
    square = Image.new("L", (side, side), 255)

    square.paste(
        crop,
        (
            (side - crop.width) // 2,
            (side - crop.height) // 2,
        ),
    )

    # Match the sketch model's 224 x 224 training-image format.
    square = square.resize((224, 224), Image.Resampling.LANCZOS)

    # The model expects an RGB image.
    return square.convert("RGB")


def predict(image: Image.Image):
    processor, model, device = load_classifier()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=min(3, probabilities.numel()),
    )

    results = []

    for score, index in zip(top_values.tolist(), top_indices.tolist()):
        label = model.config.id2label.get(index, str(index))
        results.append((label, float(score)))

    return results



with st.sidebar:
    st.subheader("Drawing settings")
    brush_size = st.slider("Brush size", min_value=4, max_value=24, value=12)
    st.caption("The pretrained classifier recognizes 345 drawing categories.")
    st.divider()
    st.caption("The model and labels are downloaded once and cached beside this app.")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("✏️ Your drawing")
    canvas_key = f"drawing_canvas_{st.session_state.get('clear_counter', 0)}"
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=brush_size,
        stroke_color="#111827",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=420,
        width=520,
        drawing_mode="freedraw",
        key=canvas_key,
        return_image_data=True,
    )
    st.caption("Draw one object in the white box. Bold, centered shapes usually work best.")
    col_a, col_b = st.columns(2)
    with col_a:
        guess_clicked = st.button("🔮 Guess My Drawing", type="primary", use_container_width=True)
    with col_b:
        clear_clicked = st.button("🧹 Clear / Start Again", use_container_width=True)

if clear_clicked:
    st.session_state["clear_counter"] = st.session_state.get("clear_counter", 0) + 1
    st.rerun()



if guess_clicked:
    try:
        if canvas_result.image_data is None:
            raise ValueError("Draw something on the canvas first.")

        sketch_image = preprocess_canvas(canvas_result.image_data)

        with right:
            st.subheader("🔍 AI input preview")
            st.image(
                sketch_image,
                caption="Processed drawing passed to the model",
                width=280,
                
            )

        with st.spinner("The AI is thinking…"):
            guesses = predict(sketch_image)

        with right:
            st.subheader("🤖 AI's guesses")
            winner, score = guesses[0]

            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#0f766e,#1d4ed8);
                            padding:1.2rem;border-radius:18px;text-align:center;">
                    <div style="font-size:1rem;color:#dbeafe;">
                        I think you drew…
                    </div>
                    <div style="font-size:2.5rem;font-weight:800;color:white;">
                        {winner.title()}
                    </div>
                    <div style="font-size:1.1rem;color:white;">
                        {score * 100:.1f}% model score
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.write("**Top 3 guesses**")

            for rank, (label, value) in enumerate(guesses, start=1):
                pct = max(0.0, min(100.0, value * 100))
                st.markdown(
                    f'<div class="guess-card"><b>#{rank} &nbsp; {label.title()}</b>'
                    f'<div style="background:#334155;border-radius:99px;height:9px;margin-top:9px;">'
                    f'<div style="background:#38bdf8;width:{pct:.1f}%;height:9px;border-radius:99px;"></div>'
                    f'</div><div style="text-align:right;color:#cbd5e1;font-size:.9rem;">'
                    f'{pct:.1f}%</div></div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "Scores are the model's output scores; a high score is not a guarantee."
            )

    except Exception as exc:
        with right:
            st.subheader("🤖 AI's guesses")
            st.warning(str(exc))
            st.info(
                "Internet is required for the first model download. After the "
                "model and labels are saved, predictions can run offline."
            )


else:
    with right:
        st.subheader("🤖 Ready to guess?")

        st.markdown(
            """
            <div class="guess-card">
                <h3>How to play</h3>
                <ol>
                    <li>Draw one object on the canvas.</li>
                    <li>Press <b>Guess My Drawing</b>.</li>
                    <li>See the AI's top three guesses.</li>
                    <li>Try to fool the AI with a tricky doodle!</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Try drawing:**")
        st.write("🍎 Apple · 🚲 Bicycle · 🐱 Cat · 🏠 House · ⭐ Star")

        st.markdown(
            '<p class="small-note">Tip: Keep the drawing centered and use bold strokes.</p>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Scores are the model's output scores; a high score is not a guarantee."
        )

st.divider()

st.markdown(
    '<p style="text-align:center;color:#cbd5e1 !important;">'
    'Built with Python · Streamlit · pretrained Quick, Draw! classifier</p>',
    unsafe_allow_html=True,
)
