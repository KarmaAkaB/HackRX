import fitz  # PyMuPDF
import requests
from io import BytesIO
from typing import List
from PIL import Image
from logger_util import setup_logger

import torch
from tqdm import tqdm
from colpali_engine import ColPali, ColPaliProcessor

logger = setup_logger(__name__)

class ColpaliParser:
    def __init__(self, 
                pdf_url: str,
                BATCH_SIZE: int = 4,
                model_name: str = "vidore/colpali-v1.3"):
        self.pdf_url = pdf_url
        self.model_name = model_name
        self.BATCH_SIZE = BATCH_SIZE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.colpali_model = ColPali.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            cache_dir="./model_cache"
        )
        self.colpali_processor = ColPaliProcessor.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir="./model_cache"
        )
        
        self.images = []
        self.pdf_bytes = None
        self.embeddings = []

    def download_pdf_to_memory(self) -> None:
        logger.info(f"Downloading PDF from: {self.pdf_url}")
        try:
            response = requests.get(self.pdf_url)
            response.raise_for_status()
            self.pdf_bytes = BytesIO(response.content)
            logger.info(f"PDF downloaded to memory ({len(response.content)} bytes).")
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            raise

    def convert_pdf_to_images(self, zoom: float = 2.0) -> None:
        if self.pdf_bytes is None:
            raise ValueError("PDF not downloaded. Call download_pdf_to_memory() first.")
        logger.info("Converting PDF pages to images using PyMuPDF...")
        try:
            with fitz.open(stream=self.pdf_bytes, filetype="pdf") as doc:
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    self.images.append(image)
                    logger.info(f"Converted page {i+1} to image.")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def generate_embeddings(self):
        if not self.images:
            raise ValueError("No images available. Call convert_pdf_to_images() first.")
        
        logger.info("Generating embeddings for images...")
        try:
            for i in tqdm(range(0, len(self.images), self.BATCH_SIZE), desc="Generating embeddings"):
                batch_images = self.images[i:i + self.BATCH_SIZE]
                inputs = self.colpali_processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.colpali_model(**inputs)
                embeddings = outputs.last_hidden_state.cpu().numpy()
                self.embeddings.extend(embeddings)
                logger.info(f"Generated embeddings for batch {i // self.BATCH_SIZE + 1}.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def run(self, view_images: bool = False):
        self.download_pdf_to_memory()
        self.convert_pdf_to_images()

        if view_images:
            for i, img in enumerate(self.images[:3]):
                img.show(title=f"Page {i+1}")
        self.generate_embeddings()
        logger.info("Parser run complete.")
        return self.embeddings

if __name__ == "__main__":
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    parser = ColpaliParser(pdf_url=url)
    images = parser.run(view_images=True)
