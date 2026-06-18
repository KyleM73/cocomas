from typing import List

import numpy as np
import multiprocessing as mp
import torch

from cocomas.path_planning import A_StarDist, Node, Grid
from circles import make_circle

def get_centroids(voronoi: np.ndarray, waypoints: np.ndarray):
    new_waypoints = np.zeros_like(waypoints)
    radii = np.zeros(waypoints.shape[0])
    for i in range(waypoints.shape[0]):
        idx = np.argwhere(voronoi==i)
        if idx.size > 0:
            c = make_circle(idx)
            new_waypoints[i] = c[:2]
            radii[i] = c[2]
        else:
            new_waypoints[i] = waypoints[i]
            radii[i] = 0
    return new_waypoints, radii

def get_waypoints(map : torch.Tensor, N : int, lidar_range : float, max_iters : int = 100):
    free_idx = torch.argwhere(map==0)
    for _ in range(max_iters):
        waypoints = free_idx[torch.randint(free_idx.size()[0],(N,))]
        print("num waypoints ",waypoints.size(0))
        waypoints, radii = eval_waypoints(map, waypoints)
        if torch.max(radii) < lidar_range:
            break
        N += 1
    print(waypoints)
    print(torch.max(radii))
    return waypoints

def eval_waypoints(map : torch.Tensor, waypoints: torch.Tensor, eps: float = 1):
    err = np.inf
    while err >= eps:
        print("eval loop")
        voronoi = get_voronoi(map, waypoints)
        new_waypoints,radii = get_centroids(voronoi, waypoints)
        err = np.linalg.norm(waypoints-new_waypoints)
        waypoints = torch.from_numpy(new_waypoints)
        print(err)
    return waypoints, torch.tensor(radii)

def get_voronoi(map: torch.Tensor, waypoints: torch.Tensor):
    with mp.Pool() as pool:
        dists = pool.map(
            eval_loop,              # fn to multiprocess
            [
                {
                    "map"      : map,
                    "waypoint" : tuple(waypoints[i].tolist()),
                    "map_h"    : map.size(0),
                    "map_w"    : map.size(1),
                } 
                for i in range(waypoints.size(0))
                ]
        )
    dists_to_waypoints = np.concatenate(dists, axis=0)
    voronoi = np.argmin(dists_to_waypoints, axis=0)
    voronoi = np.where(map,np.inf,voronoi)
    return voronoi

def eval_loop(arg_dict):
    ## arg_dict: map: torch.Tensor, waypoint: tuple, map_h: int, map_w: int
    dists = np.ones((1,arg_dict["map_h"],arg_dict["map_w"])) * np.inf
    grid = Grid(arg_dict["map"].numpy())

    tgt = Node()
    tgt.set_pose(arg_dict["waypoint"])
    for i in range(arg_dict["map_h"]):
        for j in range(arg_dict["map_w"]):
            src = Node()
            src.set_pose((i,j))
            dists[0,i,j] = A_StarDist(grid, src, tgt, dists)
    return dists
                    
if __name__ == "__main__":
    import torchvision
    #from cocomas import maps
    #map_file = "cocomas/maps/empty.png"
    #map = torchvision.io.read_image(map_file, torchvision.io.ImageReadMode.GRAY)/255
    map = torch.zeros((200,200))
    map[:20,50] = 1
    print("map ",map.size())
    #map = torch.cat([map,map],dim=0)
    get_waypoints(map, 2, 100)

