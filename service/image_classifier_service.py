import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText


model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float32, # CPUs generally prefer float32 over bfloat16
).to("cpu")
print("Loaded Model to the cpu.")


def generate_response(image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},            
            ]
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False, # We want the string, not the tensor yet
    )

    # 2. Extract the image object from your messages
    # (Assuming you have a PIL image or the URL processed)
    if image_path.startswith("http"):
        image = Image.open(requests.get(messages[0]["content"][0]["url"], stream=True).raw)
    else:
        image = Image.open(messages[0]["content"][0]["url"])

    image = image.convert("RGB")

    # 3. Use the processor to combine them—this handles the "tensor creation" correctly
    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt",
        padding=True
    ).to(model.device, dtype=torch.bfloat16)

    # 4. Generate
    generated_ids = model.generate(**inputs, max_new_tokens=64)

    input_token_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[0][input_token_len:]

    # 3. Decode only those new tokens
    response_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # 4. Construct the JSON-style dictionary
    result = {
        "user": messages[0]["content"][1]["text"], # Extracts your original prompt
        "response": response_text
    }

    # Return as a dictionary (FastAPI automatically converts this to JSON)
    return result

