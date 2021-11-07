categories = ['no object', 'hat', 'vest', 'worker']

size = (1024, 1024)
n_jobs = 0
epochs = 200
batch_size = 1
lr = 1.0
update = 1
save = 1
evaluate = 2

fpn_strides = [8, 16, 32, 64, 128, 256, 512]
reg_max = 6
max_object_per_sample = 55

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

tau = 0.9

logdir = './model/'
# logdir = '/content/drive/MyDrive/AI/AdaMatch/ObjectDetection/Ver_1.9.0/model/'
