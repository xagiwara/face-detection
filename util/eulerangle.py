class EulerAngle:
    roll: float
    pitch: float
    yaw: float

    def __init__(self, roll: float, pitch: float, yaw: float) -> None:
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


__all__ = ["EulerAngle"]
