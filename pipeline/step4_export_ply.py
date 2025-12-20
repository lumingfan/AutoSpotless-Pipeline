import torch
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

def construct_list_of_attributes(num_rest_sh_channels):
    c = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    extra = []
    for i in range(num_rest_sh_channels):
        extra.append(f'f_rest_{i}')
    
    attr = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + c + extra + \
           ['opacity'] + \
           ['scale_0', 'scale_1', 'scale_2'] + \
           ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    return attr

def export_ply(ckpt_path, output_path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 1. 确定参数字典
    if 'model' in ckpt:
        params = ckpt['model']
    elif 'splats' in ckpt:
        params = ckpt['splats']
    else:
        params = ckpt
    
    print(f"   Keys found: {list(params.keys())}")

    # 2. 智能提取 (Key Mapping)
    try:
        # --- 位置 ---
        if 'means3d' in params:
            xyz = params['means3d'].numpy()
        elif 'means' in params:
            xyz = params['means'].numpy()
        else:
            raise KeyError("Cannot find 'means3d' or 'means'")
            
        # --- 不透明度 ---
        # 有时候是 opacities (N,1) 有时候是 sigmoid 后的
        opacities = params['opacities'].numpy()
        
        # --- 缩放 ---
        scales = params['scales'].numpy()
        
        # --- 旋转 ---
        # 注意: gsplat 通常存的是 (w, x, y, z)
        rots = params['quats'].numpy() 
        
        # --- 颜色 / SH ---
        if 'sh0' in params and 'shN' in params:
            sh0 = params['sh0'].numpy()
            shN = params['shN'].numpy()
            
            # [N, 1, 3] -> [N, 3]
            features_dc = sh0.reshape(sh0.shape[0], 3)
            
            # [N, K, 3] -> [N, K*3]
            # shN 的形状通常是 [N, (degrees+1)^2 - 1, 3]
            features_extra = shN.reshape(shN.shape[0], -1)
            
        elif 'colors' in params:
            features_dc = params['colors'].numpy()
            features_extra = np.zeros((xyz.shape[0], 0))
        else:
            raise KeyError("Cannot find sh0/shN or colors")

    except Exception as e:
        print(f"❌ Error extracting parameters: {e}")
        return

    # 3. 写入 PLY
    print(f"   Constructing PLY with {xyz.shape[0]} points...")
    dtype_full = [(n, 'f4') for n in construct_list_of_attributes(features_extra.shape[1])]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = np.zeros_like(xyz[:, 0])
    elements['ny'] = np.zeros_like(xyz[:, 0])
    elements['nz'] = np.zeros_like(xyz[:, 0])
    
    elements['f_dc_0'] = features_dc[:, 0]
    elements['f_dc_1'] = features_dc[:, 1]
    elements['f_dc_2'] = features_dc[:, 2]
    
    for i in range(features_extra.shape[1]):
        elements[f'f_rest_{i}'] = features_extra[:, i]
    
    # 确保 opacity 维度正确
    elements['opacity'] = opacities.flatten()
    
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    
    elements['rot_0'] = rots[:, 0]
    elements['rot_1'] = rots[:, 1]
    elements['rot_2'] = rots[:, 2]
    elements['rot_3'] = rots[:, 3]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(str(output_path))
    
    print(f"✅ Converted .pt to .ply successfully!")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", help="Path to .pt checkpoint file")
    parser.add_argument("output_path", help="Path to save .ply file")
    args = parser.parse_args()
    
    export_ply(args.ckpt_path, args.output_path)