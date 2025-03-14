# Randomly generating different distances for training
distance_path = '/home/user/lnh_2023/Ada-Holo/dataset/distance.txt'
distance_path2 = '/home/user/lnh_2023/Ada-Holo/dataset/distance-valid.txt'
import random

def gen_flex_distance(distance_near, distance_far):
    outfile = open(distance_path, 'w+')
    for i in range(800*40):
        d = random.randint(distance_near, distance_far)
        outfile.write(str(d)+'\n')
    outfile.close()

    outfile = open(distance_path2, 'w+')
    for i in range(100*40):
        d = int((i%11)*(distance_far-distance_near)/10+distance_near)
        outfile.write(str(d)+'\n')
    outfile.close()