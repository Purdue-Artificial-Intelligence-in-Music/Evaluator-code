from BeatNet.BeatNet import BeatNet

estimator = BeatNet(1, mode='online', inference_model='DBN', plot=[], thread=False)

Output = estimator.process('correct.wav')
print(Output)
print("exits")


