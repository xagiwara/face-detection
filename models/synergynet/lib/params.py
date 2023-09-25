import os.path as osp
import numpy as np
import scipy.io as sio
from lib.synergynet.utils.io import _load
from env import DATA_SYNERGYNET
from typing import Optional

d = osp.realpath(osp.join(DATA_SYNERGYNET, "3dmm_data"))


class _ParamsPack:
    def __init__(self):
        try:
            self.keypoints = _load(osp.join(d, "keypoints_sim.npy"))

            # PCA basis for shape, expression, texture
            self.w_shp = _load(osp.join(d, "w_shp_sim.npy"))
            self.w_exp = _load(osp.join(d, "w_exp_sim.npy"))
            # param_mean and param_std are used for re-whitening
            meta = _load(osp.join(d, "param_whitening.pkl"))
            self.param_mean = meta.get("param_mean")
            self.param_std = meta.get("param_std")
            # mean values
            self.u_shp = _load(osp.join(d, "u_shp.npy"))
            self.u_exp = _load(osp.join(d, "u_exp.npy"))
            self.u = self.u_shp + self.u_exp
            self.w = np.concatenate((self.w_shp, self.w_exp), axis=1)
            # base vector for landmarks
            self.w_base = self.w[self.keypoints]
            self.w_norm = np.linalg.norm(self.w, axis=0)
            self.w_base_norm = np.linalg.norm(self.w_base, axis=0)
            self.u_base = self.u[self.keypoints].reshape(-1, 1)
            self.w_shp_base = self.w_shp[self.keypoints]
            self.w_exp_base = self.w_exp[self.keypoints]
            self.std_size = 120
            self.dim = self.w_shp.shape[0] // 3
            self.tri = sio.loadmat(osp.join(d, "tri.mat"))
        except:
            raise RuntimeError("Missing data")


_paramspack: Optional[_ParamsPack] = None


class ParamsPack:
    """Parameter package"""

    def load(self):
        global _paramspack
        if _paramspack is None:
            _paramspack = _ParamsPack()

    def get(self, name: str):
        global _paramspack
        self.load()
        return getattr(_paramspack, name)

    keypoints = property(lambda self: self.get("keypoints"))
    w_shp = property(lambda self: self.get("w_shp"))
    w_exp = property(lambda self: self.get("w_exp"))
    param_mean = property(lambda self: self.get("param_mean"))
    param_std = property(lambda self: self.get("param_std"))
    u_shp = property(lambda self: self.get("u_shp"))
    u_exp = property(lambda self: self.get("u_exp"))
    u = property(lambda self: self.get("u"))
    w = property(lambda self: self.get("w"))
    w_base = property(lambda self: self.get("w_base"))
    w_norm = property(lambda self: self.get("w_norm"))
    w_base_norm = property(lambda self: self.get("w_base_norm"))
    u_base = property(lambda self: self.get("u_base"))
    w_shp_base = property(lambda self: self.get("w_shp_base"))
    w_exp_base = property(lambda self: self.get("w_exp_base"))
    std_size = property(lambda self: self.get("std_size"))
    dim = property(lambda self: self.get("dim"))
    tri = property(lambda self: self.get("tri"))
