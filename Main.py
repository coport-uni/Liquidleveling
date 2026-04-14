"""Minimal example that drives two digital pins on the Uno R4."""

from py_arduino import PyArduino


class AILeveling:
    """Thin wrapper around PyArduino for smoke-testing digital output."""

    def __init__(self, board_type: str):
        """Connect to the board.

        Args:
            board_type: Either ``"wifi"`` or ``"minima"``.
        """
        self.pa = PyArduino(board_type)

    def run_example(self):
        """Drive pins 7 and 5 low to verify the transport works."""
        pin = 7
        self.pa.run_digital_write(pin, False)
        pin = 5
        self.pa.run_digital_write(pin, False)


def main():
    """Loop the digital-write example forever."""
    al = AILeveling("minima")
    while True:
        al.run_example()


main()
