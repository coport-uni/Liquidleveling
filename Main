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
        self.pa.run_digital_write(pin, True)
        time.sleep(2)
        self.pa.run_digital_write(pin, False)
        time.sleep(2)

        # for i in range(4):
        #     self.pa.run_digital_write(i+4, True)
        #     time.sleep(2)
        #     self.pa.run_digital_write(i+4, False)
        #     time.sleep(2)

        # value = pa.get_analog_state(2)
        # print(value)

def main():
    al = AILeveling("minima")
    while True:
        al.run_example()

if __name__ == "__main__":
    main()
        
