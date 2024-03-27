
from Listener import Listener
from Calculator import Calculator

def main():
    listener = Listener()
    buffer = listener.listen(duration=10)
    calculator = Calculator()
    df = calculator.calculate(buffer, fast = False)
    
    #NOTE Right now, yin is converting a rest to a C7!!!!
    #Fix that. C7 is just the lowest possible note.

if __name__ == "__main__":
    main()