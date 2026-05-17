import cv2
import numpy as np
from utils.constants import BLUE, BLACK


class AirCanvas:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.canvas = None
        self.color = BLUE
        self.brush_thickness = 8
        self.eraser_thickness = 30

        # smoothing factor: 0.0 = very smooth, 1.0 = no smoothing
        self.smooth_factor = 0.25
        self.smoothed_x = None
        self.smoothed_y = None

    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw(self, frame, x, y):
        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = x
            self.smoothed_y = y

        # Exponential moving average smoothing
        self.smoothed_x = int(self.smooth_factor * x + (1 - self.smooth_factor) * self.smoothed_x)
        self.smoothed_y = int(self.smooth_factor * y + (1 - self.smooth_factor) * self.smoothed_y)

        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = self.smoothed_x, self.smoothed_y

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (self.smoothed_x, self.smoothed_y),
            self.color,
            self.brush_thickness,
            lineType=cv2.LINE_AA
        )

        self.prev_x, self.prev_y = self.smoothed_x, self.smoothed_y

    def erase(self, x, y):
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (x, y),
            BLACK,
            self.eraser_thickness,
            lineType=cv2.LINE_AA
        )

        self.prev_x, self.prev_y = x, y

    def reset_position(self):
        self.prev_x, self.prev_y = 0, 0
        self.smoothed_x = None
        self.smoothed_y = None

    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas[:] = 0

    def merge_canvas(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        frame_bg = cv2.bitwise_and(frame, mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, mask)

        return cv2.add(frame_bg, canvas_fg)