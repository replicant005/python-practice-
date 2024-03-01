import os
import subprocess

def read_input():
    return input("$ ")

def parse_input(input_str):
    return input_str.split()

def execute_command(command):
    try:
        subprocess.run(command)
    except FileNotFoundError:
        print(f"Command not found: {command[0]}")

def main():
    while True:
        input_str = read_input()
        command = parse_input(input_str)
        execute_command(command)

if __name__ == "__main__":
    main()
