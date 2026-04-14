import sys
import time
import numpy as np
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi import telemetrix_uno_r4_wifi
from telemetrix_uno_r4.minima.telemetrix_uno_r4_minima import telemetrix_uno_r4_minima

import logging

class PyArduino():
    def __init__(self, board_type : str):
        """
        This function initialize board as board_type. User must input wifi, minima

        Input : Str
        Output : None
        """
        if board_type == "wifi":
            self.board = telemetrix_uno_r4_wifi.TelemetrixUnoR4WiFi(transport_type=1)
            print("selected_wifi")

        elif board_type == "minima":
            self.board = telemetrix_uno_r4_minima.TelemetrixUnoR4Minima()
            print("selected_minima")   

        else:
            print("error")

    def run_digital_write(self, pin_number : int, pin_state : bool):
        """
        This function write arduino digital pins. Double checking with 13 led is recommended.

        Input : int, bool
        Output : None
        """
        self.board.set_pin_mode_digital_output(pin_number)

        if pin_state is True:
            self.board.digital_write(pin_number, 1)
            logging.info(f'{pin_number} is {pin_state}')

        elif pin_state is False:
            self.board.digital_write(pin_number, 0)
            logging.info(f'{pin_number} is {pin_state}')
            
        time.sleep(0.001)

        # self.board.shutdown()
    
    def get_digital_state(self, pin_number : int):
        """
        This function read arduino digital pins. Double checking with any pin is recommended.

        Input : int
        Output : int
        """
        global digital_value
        digital_value = None
        self.board.set_pin_mode_digital_input_pullup(pin_number, callback = self._get_digital_state_slicer)
        
        while digital_value is None:
            time.sleep(0.01)

        # self.board.shutdown()

        return digital_value

    def run_digital_pwm_write(self, pin_number : int, pwm_intensity_percentage : int):
        max_value = 256
        self.board.set_pin_mode_analog_output(pin_number)
        self.board.analog_write(pin_number, round(pwm_intensity_percentage / 100 * max_value))
        time.sleep(.005)
        # self.board.set_pin_mode_digital_output(pin_number)

    def get_analog_state(self, pin_number : int):
        """
        This function read arduino analog pins. Double checking with any pin is recommended.

        Input : int
        Output : int
        """
        global analog_value

        analog_value = None
        self.board.set_pin_mode_analog_input(pin_number, differential = 0, callback = self._get_analog_state_slicer)
        while analog_value is None:
            time.sleep(0.01)

        return analog_value
    
    def run_pump_speed(self, target_speed: int):
        """
        This function simplify usaage of MCP4728 controller. Max speed is limited to 20.

        Input : int
        """
        speed = np.clip(target_speed, 0, 20)
        self._run_MCP4728_control(1, speed)
        
    def _get_digital_state_slicer(self, data):
        """
        This internal function slicing arduino's output datastream.

        Input : str
        Output : int
        """
        global digital_value
        pin_mode_index = 0
        pin_number_index = 1
        pin_state_index = 2

        digital_value = data[pin_state_index]

    def _get_analog_state_slicer(self, data):
        """
        This internal function slicing arduino's output datastream.

        Input : str
        Output : int
        """
        global analog_value
        pin_mode_index = 0
        pin_number_index = 1
        pin_state_index = 2

        # print(data[pin_state_index])

        analog_value = data[pin_state_index]

    def _run_MCP4728_control(self, pin_number : int, voltage_percentage : int):
        '''
        This function communicate MCP4728 with IIC communication. you may need to update your repo to newest version.

        Input : int, int
        '''
        max_value = 255
        voltage_factor = 5 / 3.3
        error_compensate_percentage = voltage_percentage / 50
        self.board.MCP4728_control(pin_number, round((voltage_percentage + error_compensate_percentage) / 100 * max_value * voltage_factor))
        time.sleep(.005)

def main():
    """
    This function holds example main flow

    Input : None
    Output : None
    """
    pa = PyArduino("minima")
    while True:
        pa.run_pump_speed(10)
        time.sleep(3)
        pa.run_pump_speed(10)
        time.sleep(3)
        # pa._run_MCP4728_control(1, 20)
        # time.sleep(3)
        # pa._run_MCP4728_control(1, 60)
        # time.sleep(3)
    # pin_number_power = 10
    # pin_number_rpm = 11
    # pa.run_digital_write(pin_number_power, False)
    # while True:
    #     pa.run_digital_pwm_write(pin_number_rpm, 100)
    #     time.sleep(3)
    #     pa.run_digital_pwm_write(pin_number_rpm, 0)
    #     time.sleep(3)
    # 센서 예제
    # while True:
    #     value0 = pa.get_analog_state(0)
    #     print("A0 is " + str(value0))
    #     value1 = pa.get_analog_state(1)
    #     print("A1 is " + str(value1))
    # while True:
    #     value = pa.get_digital_state(8)
    #     print(value)

    # 디지털 입출력 예제
    # for pin_number in range(4,8):
    #     pa.run_digital_write(pin_number, True)
    #     time.sleep(1)
    #     pa.run_digital_write(pin_number, False)

    # 밸브, 외란 펌프 예제
    # pa.run_digital_write(4, True)
    # pa.run_digital_write(5, True)
    
    # pa.run_digital_write(6, True)
    # pa.run_digital_write(7, True)


if __name__ == '__main__':
    main()
