import pytesseract
from typing import List

def apply_ocr(image):
  data = pytesseract.image_to_data(image, output_type="dict")
  words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

  # filter empty words and corresponding coordinates
  irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
  words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
  left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
  top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
  width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
  height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

  # turn coordinates into (left, top, left+width, top+height) format
  actual_boxes = [[x, y, x + w, y + h] for x, y, w, h in zip(left, top, width, height)]

  return words, actual_boxes

def _generate_document_output(image, words_by_page, boxes_by_page):

        processed_pages = []
        for words, boxes in zip(words_by_page, boxes_by_page):
            width, height = image.width, image.height

            normalized_boxes = [
                [
                    max(min(c, 1000), 0)
                    for c in [
                        int(1000 * (box[0] / width)),
                        int(1000 * (box[1] / height)),
                        int(1000 * (box[2] / width)),
                        int(1000 * (box[3] / height)),
                    ]
                ]
                for box in boxes
            ] 
            word_boxes = [x for x in zip(words, normalized_boxes)]
            processed_pages.append((image, word_boxes))

        return {"image": processed_pages}
