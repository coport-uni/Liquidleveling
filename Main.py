from PyArduino import PyArduino
import time

class AILeveling():
    def __init__(self, board_type : str):
        """
        This function initialize board as board_type. User must input wifi, minima

        Input : Str
        Output : None
        """
        self.pa = PyArduino(board_type)
    
    def run_example(self):
        """
        This function command arduino with digital input or analog output

        Input : None
        Output : None
        """
        pin = 7
        self.pa.run_digital_write(pin, False)
        pin = 5
        self.pa.run_digital_write(pin, False)

def main():
    al = AILeveling("minima")
    while True:
        al.run_example()


main()
