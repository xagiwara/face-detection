from math import floor, ceil
import cv2


class CropRect:
    top: float
    left: float
    bottom: float
    right: float

    def __init__(self, top: float, left: float, bottom: float, right: float) -> None:
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def crop(self, image):
        height, width = image.shape[:2]
        top = floor(self.top)
        left = floor(self.left)
        bottom = ceil(self.bottom)
        right = ceil(self.right)

        if bottom > height or right > width:
            image = cv2.copyMakeBorder(
                image,
                0,
                max(bottom - height, 0),
                0,
                max(right - width, 0),
                cv2.BORDER_CONSTANT,
                (0, 0, 0),
            )

        if top < 0:
            image = cv2.copyMakeBorder(
                image, -top, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
            )
            top = 0
            bottom += -top

        if left < 0:
            image = cv2.copyMakeBorder(
                image, 0, 0, -left, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
            )
            left = 0
            right += -left

        return image[top:bottom, left:right]

    def square_expand(self):
        cx = (self.left + self.right) / 2
        cy = (self.top + self.bottom) / 2
        halfsize = max(self.right - self.left, self.bottom - self.top) / 2
        return CropRect(cy - halfsize, cx - halfsize, cy + halfsize, cx + halfsize)

    def square_shrink(self):
        cx = (self.left + self.right) / 2
        cy = (self.top + self.bottom) / 2
        halfsize = min(self.right - self.left, self.bottom - self.top) / 2
        return CropRect(cy - halfsize, cx - halfsize, cy + halfsize, cx + halfsize)


__all__ = ["CropRect"]
