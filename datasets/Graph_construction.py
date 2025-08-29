import os
import numpy as np
from skimage import io, measure
#from skimage.segmentation import find_boundaries  # 不再使用原函数
from scipy import sparse
import networkx as nx
from tqdm import tqdm

# 内存优化配置
DTYPE_IMG = np.uint16    # 用于存储图像数据
DTYPE_LABEL = np.uint16  # 用于标签数据
DTYPE_FLOAT = np.float32 # 必须使用浮点时使用的类型

# 自定义边界检测函数，利用相邻像元差异计算边界
def custom_find_boundaries(label_img):
    # 创建与输入同大小的布尔数组（1字节/元素），内存占用较低
    boundaries = np.zeros(label_img.shape, dtype=bool)
    # 垂直方向比较：下方与上方
    boundaries[1:, :] |= (label_img[1:, :] != label_img[:-1, :])
    boundaries[:-1, :] |= (label_img[:-1, :] != label_img[1:, :])
    # 水平方向比较：右侧与左侧
    boundaries[:, 1:] |= (label_img[:, 1:] != label_img[:, :-1])
    boundaries[:, :-1] |= (label_img[:, :-1] != label_img[:, 1:])
    return boundaries

# %% File 1: Main Script
def main_processing():
    # Input parameters
    fn_ID = 'Subset1'
    input_folder = r'F:\Exp3\Subset1'
    mat_folder = os.path.join(input_folder, 'Mat')

    # File paths
    D_path = os.path.join(input_folder, f'{fn_ID}_VHR_RGB.tif')
    bldMap_path_255 = os.path.join(input_folder, f'{fn_ID}_binary.tif')
    img2_path = os.path.join(input_folder, f'{fn_ID}_seg_osm.tif')
    luwang_path = os.path.join(input_folder, f'{fn_ID}_luwang.tif')
    label_path = r'F:\data_bd_type\data_raw\image01_label.tif'

    # 优化后的数据加载（按需转换数据类型）
    def load_image(path, dtype, default_dtype=np.uint8):
        img = io.imread(path).astype(default_dtype)
        return img.astype(dtype) if dtype != default_dtype else img

    # 加载数据时指定数据类型
    label = load_image(label_path, DTYPE_LABEL)
    bld_img = load_image(bldMap_path_255, DTYPE_IMG, np.uint8)  # 二值图像使用uint8
    D = load_image(D_path, DTYPE_FLOAT)  # 深度图可能需要浮点
    img2 = load_image(img2_path, DTYPE_IMG) + 1
    luwang = load_image(luwang_path, DTYPE_IMG, np.uint8)

    # Merge small objects
    regPXL = measure.regionprops(img2.astype(np.uint16))
    num_obj = len(regPXL)
    img2_merge = img2.view()
    a_th = 9

    obj_lb = derive_obj_lb(img2, bld_img)
    # 调用自定义边界函数替换 find_boundaries
    edg = edgeList_via_bd_fast(img2)[0]
    G = nx.DiGraph()
    G.add_edges_from(edg)
    adj = nx.adjacency_matrix(G).astype(bool).astype(int)

    for i in tqdm(range(num_obj), desc='Processing objects'):
        a_i = regPXL[i].area
        if a_i < a_th:
            neigh_i = adj[i].indices
            num_neigh = len(neigh_i)
            area_nb_i = np.zeros(num_neigh)

            for j in range(num_neigh):
                area_nb_i[j] = regPXL[neigh_i[j]].area

            neig_lb_i = obj_lb[neigh_i]
            lb_i_vec = np.full(num_neigh, obj_lb[i])
            small_nb_i = (neig_lb_i == lb_i_vec).astype(float)
            ix_i = np.argmax(area_nb_i * small_nb_i)
            pxl_i = regPXL[i].coords
            img2_merge[pxl_i[:, 0], pxl_i[:, 1]] = neigh_i[ix_i]

    # Refine and save results
    img2_refined = reOrdSegID_fast_li(img2_merge, 1)
    np.save(os.path.join(mat_folder, f'{fn_ID}_img2_refined.npy'), img2_refined)

    # Generate graph structures
    edg, edg_cell = edgeList_via_bd_fast(img2_refined)
    reg_cls = derive_obj_lb(img2_refined, label)
    reg_bd = derive_obj_lb(img2_refined, bld_img)
    reg_luwang = derive_obj_lb(img2_refined, luwang)

    G = nx.DiGraph()
    G.add_edges_from(edg)
    adj = nx.adjacency_matrix(G)

    # Create adjacency matrices
    mask_sameRoad = (reg_luwang[:, None] == reg_luwang[None, :])
    adj_osm = adj.multiply(mask_sameRoad)

    # Weighted adjacency
    w1, w2 = 1, 1
    adj_weighted = w1 * adj + w2 * adj_osm

    # Save outputs
    sparse.save_npz(os.path.join(mat_folder, f'{fn_ID}_graph_base_adj.npz'), adj)
    sparse.save_npz(os.path.join(mat_folder, f'{fn_ID}_graph_base_adj_osm.npz'), adj_osm)
    sparse.save_npz(os.path.join(mat_folder, f'{fn_ID}_graph_base_adj_weighted.npz'), adj_weighted)

# %% File 2: Helper Functions
def derive_obj_lb(img2, lcMap):
    regions = measure.regionprops(img2.astype(np.int16))
    obj_lb = np.zeros(len(regions), dtype=np.int16)

    for i, region in enumerate(regions):
        # 如果区域为空，则直接赋予默认标签0，避免后续运算出错
        if region.area == 0:
            obj_lb[i] = 0
            continue
        try:
            coords = region.coords
        except ValueError:
            obj_lb[i] = 0
            continue

        if coords.size == 0:
            obj_lb[i] = 0
            continue

        # 根据区域内在lcMap中的像元值来确定该区域的标签
        labels = lcMap[tuple(coords.T)]
        unique_labels, counts = np.unique(labels, return_counts=True)
        obj_lb[i] = unique_labels[np.argmax(counts)]

    return obj_lb


def edgeList_via_bd_fast(img2):
    img2_uint16 = img2.astype(np.uint16)  # 确保数据类型正确
    # 使用自定义边界检测函数
    img_bd = custom_find_boundaries(img2_uint16)
    r, c = img2_uint16.shape
    regions = measure.regionprops(img2_uint16)
    edg_cell = [[] for _ in range(len(regions))]

    with tqdm(total=len(regions), desc='Generating edges') as pbar:
        for i, region in enumerate(regions):
            coords = region.coords
            mask = img_bd[coords[:, 0], coords[:, 1]]
            boundary_coords = coords[mask]

            neighbors = set()
            for (y, x) in boundary_coords:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < r and 0 <= nx < c:
                            neighbor_id = img2[ny, nx]
                            if neighbor_id != (i + 1):  # Python uses 0-based indexing
                                neighbors.add(neighbor_id - 1)  # Adjust indexing
            edg_cell[i] = [(i, n) for n in neighbors if n != i]
            pbar.update(1)

    edg = [edge for sublist in edg_cell for edge in sublist]
    return np.array(edg), edg_cell

def reOrdSegID_fast_li(img_raw, reorder_type):
    unique_vals = np.unique(img_raw)
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    img_new = np.vectorize(mapping.get)(img_raw)
    return img_new - (1 if reorder_type == 0 else 0)

if __name__ == "__main__":
    main_processing()
