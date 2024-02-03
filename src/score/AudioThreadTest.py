from AudioThreadWithBufferPorted import *

from librosa import *
import matplotlib.pyplot as plt


def test(arg):
    #print(type(arg[0]))
    print("arg: ", arg)
    #numpy_array = np.frombuffer(arg, dtype=np.float64)
    f0, voiced_flag, voiced_probs = pyin(y=arg,
                                             fmin=note_to_hz('C2'),
                                             fmax=note_to_hz('C7'))
    times = times_like(f0)
    print("librosa out: ", f0)

    return
def main():
    my_thread = AudioThreadWithBufferPorted('my thread', rate=44100, starting_chunk_size=1024, process_func=test)
    try: 
        my_thread.start()
        print("Start thread")
        while True:
            print(my_thread.input_on)
            time.sleep(1.5)
    except KeyboardInterrupt:
        my_thread.stop_request = True

if __name__ == "__main__":
    main()