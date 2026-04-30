import argparse
import serial
PORT = "COM3"

COMMANDS = {
    "0": "120,1,120,1\n",  # Forward slow
    "1": "120,0,120,0\n",  # Backward slow
    "2": "120,0,120,1\n",  # Rotate left slow
    "3": "120,1,120,0\n",  # Rotate right slow
    "4": "0,1,0,1\n",      # Stop
}


def send_command(ser, command):
    ser.write(COMMANDS[command].encode("utf-8"))
    print(f"Sent: {COMMANDS[command].strip()}")


def main():
    parser = argparse.ArgumentParser(description="Remote control for Bluetooth car.")
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port for the Bluetooth connection (default: COM8)",
    )
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, 9600, timeout=1)

        print("0: Drive forward")
        print("1: Drive backward")
        print("2: Rotate left")
        print("3: Rotate right")
        print("4: Stop")
        print("5: Disconnect\n")

        while True:
            command = input("Enter your command: ")

            if command == "5":
                break

            if command not in COMMANDS:
                print("Invalid command. Try again (0-5)")
                continue

            msg = COMMANDS[command]
            ser.write(msg.encode("utf-8"))
            print(f"Sent: {msg.strip()}")

        # Stop car before disconnecting
        ser.write(COMMANDS["4"].encode("utf-8"))

    except serial.SerialException as e:
        print(f"Serial error: {e}")

    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            print("Disconnected.")


if __name__ == "__main__":
    main()
