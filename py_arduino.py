"""Thin wrapper over the Telemetrix Uno R4 client.

Exposes ``PyArduino``, a small facade around Arduino digital and analog
I/O plus an MCP4728 DAC channel used to drive the liquid-level pump.
The Telemetrix client reports pin reads through an asynchronous callback,
so the read helpers intentionally publish values through module globals
rather than instance state; the callback lifecycle makes instance
attribution awkward here.
"""

import logging
import time

import numpy as np
from telemetrix_uno_r4.minima.telemetrix_uno_r4_minima import (
    telemetrix_uno_r4_minima,
)
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi import (
    telemetrix_uno_r4_wifi,
)

# Module-level rendezvous slots written by Telemetrix async callbacks
# and read back by the synchronous getters below. See module docstring.
digital_value = None
analog_value = None


class PyArduino:
    """Facade for the Arduino Uno R4 used by the liquid-level rig.

    Wraps digital/analog I/O and an MCP4728 DAC channel so the control
    scripts can drive the pump without learning the Telemetrix
    transport directly.
    """

    def __init__(self, board_type: str):
        """Connect to the selected Uno R4 transport.

        Args:
            board_type: Either ``"wifi"`` or ``"minima"``.

        Raises:
            ValueError: If ``board_type`` is not a supported transport.
                Fail-fast avoids running later calls against an
                uninitialised ``self.board``.
        """
        if board_type == "wifi":
            self.board = telemetrix_uno_r4_wifi.TelemetrixUnoR4WiFi(
                transport_type=1,
            )
            print("selected wifi")
        elif board_type == "minima":
            self.board = telemetrix_uno_r4_minima.TelemetrixUnoR4Minima()
            print("selected minima")
        else:
            raise ValueError(f"unknown board_type: {board_type!r}")

    def run_digital_write(self, pin_number: int, pin_state: bool):
        """Drive a digital output pin high or low.

        Args:
            pin_number: Board pin index to drive.
            pin_state: ``True`` for HIGH, ``False`` for LOW.
        """
        self.board.set_pin_mode_digital_output(pin_number)

        if pin_state is True:
            self.board.digital_write(pin_number, 1)
            logging.info(f"{pin_number} is {pin_state}")
        elif pin_state is False:
            self.board.digital_write(pin_number, 0)
            logging.info(f"{pin_number} is {pin_state}")

        time.sleep(0.001)

    def get_digital_state(self, pin_number: int) -> int:
        """Read a digital input pin with the pull-up resistor enabled.

        Blocks until the Telemetrix callback publishes a value.

        Args:
            pin_number: Board pin index to sample.

        Returns:
            The last digital level reported by the board (0 or 1).
        """
        global digital_value
        digital_value = None
        self.board.set_pin_mode_digital_input_pullup(
            pin_number,
            callback=self._get_digital_state_slicer,
        )

        while digital_value is None:
            time.sleep(0.01)

        return digital_value

    def run_digital_pwm_write(
        self,
        pin_number: int,
        pwm_intensity_percentage: int,
    ):
        """Write a PWM duty cycle to an analog-capable pin.

        Args:
            pin_number: Board pin index to drive.
            pwm_intensity_percentage: Duty cycle expressed as 0-100 %.
        """
        max_value = 256
        self.board.set_pin_mode_analog_output(pin_number)
        self.board.analog_write(
            pin_number,
            round(pwm_intensity_percentage / 100 * max_value),
        )
        time.sleep(0.005)

    def get_analog_state(self, pin_number: int) -> int:
        """Read an analog input pin.

        Blocks until the Telemetrix callback publishes a value.

        Args:
            pin_number: Analog pin index to sample.

        Returns:
            The last ADC reading reported by the board.
        """
        global analog_value

        analog_value = None
        self.board.set_pin_mode_analog_input(
            pin_number,
            differential=0,
            callback=self._get_analog_state_slicer,
        )
        while analog_value is None:
            time.sleep(0.01)

        return analog_value

    def run_pump_speed(self, target_speed: int):
        """Set pump speed via the MCP4728 DAC channel.

        The useful range is 0-20; larger values cavitate the pump and
        are clamped defensively here.

        Args:
            target_speed: Requested pump speed in the 0-20 range.
        """
        speed = np.clip(target_speed, 0, 20)
        self._run_mcp4728_control(1, speed)

    def _get_digital_state_slicer(self, data):
        """Telemetrix callback: publish a digital read to the global."""
        global digital_value
        pin_state_index = 2
        digital_value = data[pin_state_index]

    def _get_analog_state_slicer(self, data):
        """Telemetrix callback: publish an analog read to the global."""
        global analog_value
        pin_state_index = 2
        analog_value = data[pin_state_index]

    def _run_mcp4728_control(
        self,
        pin_number: int,
        voltage_percentage: int,
    ):
        """Drive an MCP4728 DAC channel over I2C.

        The 5/3.3 voltage factor and the ``voltage_percentage / 50``
        error compensation term together correct for a 3.3 V MCU
        driving a 5 V DAC rail whose transfer curve is not perfectly
        linear; this empirical correction keeps the pump command
        roughly proportional to ``voltage_percentage``. Removing
        either term desynchronises the control-loop gains.

        Args:
            pin_number: MCP4728 channel index.
            voltage_percentage: Target output expressed as 0-100 %.
        """
        max_value = 255
        voltage_factor = 5 / 3.3
        error_compensate_percentage = voltage_percentage / 50
        self.board.MCP4728_control(
            pin_number,
            round(
                (voltage_percentage + error_compensate_percentage)
                / 100
                * max_value
                * voltage_factor,
            ),
        )
        time.sleep(0.005)


def main():
    """Run a minimal pump-speed loop for manual hardware smoke tests."""
    pa = PyArduino("minima")
    while True:
        pa.run_pump_speed(10)
        time.sleep(3)
        pa.run_pump_speed(10)
        time.sleep(3)


if __name__ == "__main__":
    main()
