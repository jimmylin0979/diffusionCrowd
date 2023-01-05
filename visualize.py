#
import torch
import numpy as np
import argparse

# 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main(filepath):
    
    # 
    densityMap = torch.load(filepath)
    densityMap = densityMap.cpu().numpy()
    densityMap = densityMap[0][0]

    # 
    print(np.min(densityMap), np.max(densityMap))

    #
    plt.imshow(densityMap, cmap=cm.jet)
    plt.savefig('densityMap.png')

#
if __name__ == "__main__":
    
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path of file")
    args = parser.parse_args()

    #
    main(args.file)