import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import torch

from yolov5.utils.general import xyxy2xywh, clip_coords
from geometry_utils import explode_xy, shoelace_area
from ocr_preprocessing import OCRPreprocessor


class OCR:

    def __init__(self):
        self.preprocessor = OCRPreprocessor()
        self.easyocr_reader = easyocr.Reader(['en'])

    def get_box_img(self, xyxy, imc, gain=1.02, pad=10, BGR=False):
        xyxy = torch.tensor(xyxy).view(-1, 4)
        b = xyxy2xywh(xyxy)
        b[:, 2:] = b[:, 2:] * gain + pad
        xyxy = xywh2xyxy(b).long()
        clip_coords(xyxy, imc.shape)
        crop = imc[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
        return crop

    def get_most_relevant_by_area(self, information):
        summary = {}
        for i in information:
            xy = i[0]
            xy_e = explode_xy(xy)
            area = shoelace_area(xy_e[0], xy_e[1])
            summary[i[1]] = area

        return max(summary, key=summary.get)

    def run_easy_ocr(self, box):
        extractedInformation = self.easyocr_reader.readtext(box, detail=1)
        label = self.get_most_relevant_by_area(extractedInformation)
        return label
