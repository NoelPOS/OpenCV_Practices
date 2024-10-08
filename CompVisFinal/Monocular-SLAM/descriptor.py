import numpy as np

class Descriptor(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None

    def optimize(self):
        err = optimize(self.frames, self.points, local_window, fix_points, verbose, rounds)
        culled_pt_count = 0
        for p in self.points:
            errs = []
            for f, idx in zip(p.frames, p.idxs):
                uv = f.kps[idx]
                proj = np.dot(f.pose[:3], p.homogeneous())
                proj = proj[0:2] / proj[2]
                errs.append(np.linalg.norm(proj - uv))
            if np.mean(errs) > CULLING_ERR_THRES:
                culled_pt_count += 1
                self.points.remove(p)
        return err

    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p[:3])  # Only use first 3 coordinates of the 4D point
        print("Poses:", np.array(poses))
        print("Points:", np.array(pts))
