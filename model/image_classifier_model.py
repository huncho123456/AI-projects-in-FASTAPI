from pydantic import BaseModel


class PromptModel(BaseModel):
    image_path: str
    prompt: str