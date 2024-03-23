
from Listener import Listener
from Calculator import Calculator

def main():
    print("main")
    listener = Listener()
    buffer = listener.listen(duration=5)
    print(buffer, len(buffer))
    calculator = Calculator()
    df = calculator.calculate(buffer)
    print(df)

if __name__ == "__main__":
    main()