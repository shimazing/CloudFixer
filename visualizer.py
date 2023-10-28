import os
import glob
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def save_xyz_file(path, positions, id_from=0,
        name='pointcloud', node_mask=None, n_nodes=1024):
    try:
        os.makedirs(path)
    except OSError:
        pass

    batch_size = positions.shape[0]

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [n_nodes]  * batch_size

    for batch_i in range(batch_size):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        for atom_i in range(n_nodes):
            atom = 'o'
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()


def load_xyz(file):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files


def visualize(path, max_num=25, wandb=None, postfix=''):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        positions = load_xyz(file)
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]
        print("Average distance between atoms", dists.mean().item())

        if wandb is not None:
            path = file[:-4] + '.png'
            # Log image(s)
            obj3d = wandb.Object3D({
                "type": "lidar/beta",
                "points": positions.cpu().numpy().reshape(-1, 3),
                "boxes": np.array(
                    [
                        {
                            "corners": (np.array([
                                [-1, -1, -1],
                                [-1, 1, -1],
                                [-1, -1, 1],
                                [1, -1, -1],
                                [1, 1, -1],
                                [-1, 1, 1],
                                [1, -1, 1],
                                [1, 1, 1]
                                ])*3).tolist(),
                            "label": "Box",
                            "color": [123, 321, 111], # ???
                        }
                    ]
                ),
                })
            wandb.log({'3d' + postfix: obj3d})


def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, elev=30, azim=225):
    r''' Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=elev, azim=azim)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_pointcloud_batch(path, pointclouds, pred_labels, labels, categories, vis_label=False, target=None,  elev=30, azim=225):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)
    for idx, pc in enumerate(pointclouds):
        if vis_label:
            label = categories[labels[idx].item()]
            pred = categories[pred_labels[idx]]
            colour = 'g' if label == pred else 'r'
        elif target is None:
            colour = 'g'
        else:
            colour = target[idx]
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        if vis_label:
            ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))
    plt.savefig(path)
    plt.close(fig)


def write_pc(point_cloud, output_path):
    import open3d as o3d
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(output_path, point_cloud_o3d)



################################################################################
def plot_points(output, output_name=None):
    from PIL import Image
    from datetime import datetime
    # from utils.data_helper import normalize_point_clouds
    # output = output.cpu()
    input_list = []
    for idx in range(len(output)):
        pts = output[idx]
        # pts = normalize_point_clouds([pts])
        input_img = visualize_point_clouds_3d([pts], ['out#%d' % idx])
        input_list.append(input_img)
    input_list = np.concatenate(input_list, axis=2)
    img = Image.fromarray(input_list[:3].astype(np.uint8).transpose((1, 2, 0)))
    if output_name is None:
        output_dir = './results/nv_demos/lion/'
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir, datetime.now().strftime("%y%m%d_%H%M%S.png"))
    img.save(output_name)
    # print(f'INFO save output img as {output_name}')
    return output_name


def visualize_point_clouds_3d_list(pcl_lst, title_lst, vis_order, vis_2D, bound, S):
    t_list = []
    for i in range(len(pcl_lst)):
        img = visualize_point_clouds_3d([pcl_lst[i]], [title_lst[i]] if title_lst is not None else None,
                                        vis_order, vis_2D, bound, S)
        t_list.append(img)
    img = np.concatenate(t_list, axis=2)
    return img


def visualize_point_clouds_3d(pcl_lst, title_lst=None,
                              vis_order=[2, 0, 1], vis_2D=1, bound=1.5, S=3, rgba=0):
    """
    Copied and modified from https://github.com/stevenygd/PointFlow/blob/b7a9216ffcd2af49b24078156924de025c4dbfb6/utils.py#L109 

    Args: 
        pcl_lst: list of tensor, len $L$ = num of point sets, 
            each tensor in shape (N,3), range in [-1,1] 
    Returns: 
        image with $L$ column 
    """
    # assert(type(pcl_lst) == list and torch.is_tensor(pcl_lst[0])
    #        ), f'expect list of tensor, get {type(pcl_lst)} and {type(pcl_lst[0])}'
    # if len(pcl_lst) > 1:
    #     return visualize_point_clouds_3d_list(pcl_lst, title_lst, vis_order, vis_2D, bound, S)

    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    print(f"len(pcl_lst): {len(pcl_lst)}")
    print(f"len(title_lst): {len(title_lst)}")

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    num_col = len(pcl_lst)
    assert(num_col == len(title_lst)
           ), f'require same len, get {num_col} and {len(title_lst)}'
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):

        print(f"pts.shape: {pts.shape}")
        ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
        ax1.set_title(title)
        rgb = None
        if type(S) is list:
            psize = S[idx]
        else:
            psize = S
        ax1.scatter(pts[:, vis_order[0]], pts[:, vis_order[1]],
                    pts[:, vis_order[2]], s=psize, c=rgb)
        ax1.set_xlim(-bound, bound)
        ax1.set_ylim(-bound, bound)
        ax1.set_zlim(-bound, bound)
        ax1.grid(False)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = fig2data(fig)
    res = np.transpose(res, (2, 0, 1))  # 3,H,W

    plt.close()

    if vis_2D:
        v1 = 0.5
        v2 = 0
        fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
        num_col = len(pcl_lst)
        assert(num_col == len(title_lst)
               ), f'require same len, get {num_col} and {len(title_lst)}'
        for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
            ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
            rgb = None
            if type(S) is list:
                psize = S[idx]
            else:
                psize = S
            ax1.scatter(pts[:, vis_order[0]], pts[:, vis_order[1]],
                        pts[:, vis_order[2]], s=psize, c=rgb)
            ax1.set_xlim(-bound, bound)
            ax1.set_ylim(-bound, bound)
            ax1.set_zlim(-bound, bound)
            ax1.grid(False)
            ax1.set_title(title + '-2D')
            ax1.view_init(v1, v2)  # 0.5, 0)

        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        # res_2d = np.array(fig.canvas.renderer._renderer)
        res_2d = fig2data(fig)
        res_2d = np.transpose(res_2d, (2, 0, 1))
        plt.close()

        res = np.concatenate([res, res_2d], axis=1)
    return res


def fig2data(fig):
    """
    Adapted from https://stackoverflow.com/questions/55703105/convert-matplotlib-figure-to-numpy-array-of-same-shape 
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    ## fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf



from PIL import Image
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")



xml_head_segment = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{},{},{}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="{}"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{}"/>
            <integer name="height" value="{}"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

# I also use a smaller point size
xml_ball_segment = ['']*10
xml_ball_segment[0] = \
    """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
xml_ball_segment[1] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="plastic" >
              <float name="intIOR" value="2.0"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""


xml_ball_segment[2] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="plastic" >
              <float name="intIOR" value="1.9"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""

xml_ball_segment[3] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic" >
              <float name="intIOR" value="1.9"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""
xml_ball_segment[4] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic" >
              <float name="intIOR" value="1.6"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""
xml_ball_segment[5] = \
    """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic">
              <float name="intIOR" value="1.7"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""
xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-1,1,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

import time
random_str = hex(int(time.time() + 12345))[2:] 


# replaced by command line arguments
def standardize_bbox_based_on(pcl, eps):
    pcl = pcl.numpy()[:, [0,2,1]] 
    eps = eps.numpy()[:, [0,2,1]]
    pcl, center, scale = standardize_bbox(pcl, return_center_scale=1)
    eps = (eps - center) / scale if eps is not None else None 
    offset = - 0.475 - pcl[:,2].min()
    eps[:,2] += offset 
    return torch.from_numpy(eps) 



def standardize_bbox(pcl, return_center_scale=0):
    #pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    #np.random.shuffle(pt_indices)
    #pcl = pcl[pt_indices]  # n by 3
    if torch.is_tensor(pcl):
        pcl = pcl.numpy()
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    #print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    if return_center_scale:
        return result, center, scale 
    return result



def colormap(x, y, z):
    if torch.is_tensor(x): 
        x = x.cpu().numpy()
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]



def pts2png(input_pts, file_name, colorm=[24,107,239], 
        skip_if_exists=False, is_color_list=False, 
        sample_count=256, out_width=1600, out_height=1200, 
        ball_size=0.025, do_standardize=0, same_computed_loc_color=0, material_id=0, precomputed_color=None,
        output_xml_file=None,
        use_loc_color=False, lookat_1=3, lookat_2=3, lookat_3=3, do_transform=1, trim_img=0):
    """
    Argus: 
        input_pts: (B,N,3) the points to be render 
        file_name: list; output image name 
    """
    assert(len(input_pts.shape) == 3), f'expect: B,N,3; get: {input_pts.shape}'
    assert(type(file_name) is list), f'require file_name as list'
    xml_head = xml_head_segment.format(
            lookat_1, lookat_2, lookat_3,
            sample_count, out_width, out_height)
    input_pts = input_pts.cpu()
    # print('get shape; ', input_pts.shape)
    color_list = []
    for pcli in range(0, input_pts.shape[0]):
        xmlFile = 'xml/tmp/tmp_%s.xml'%random_str if output_xml_file is None else output_xml_file 
        # ("%s/xml/%s.xml" % (folder,    filename))
        exrFile = 'xml/tmp/tmp_%s.exr'%random_str ##("%s/exr/%s.exr" % (folder, filename))
        png = file_name[pcli] 
        if skip_if_exists and os.path.exists(png):
            print(f'find png: {png}, skip ')
            continue 
        pcl = input_pts[pcli, :, :]
        if do_transform:
            pcl = standardize_bbox(pcl)
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            offset = - 0.475 - pcl[:,2].min()
            pcl[:,2] += offset 
        if do_standardize:
            pcl = standardize_bbox(pcl)
            offset = - 0.475 - pcl[:,2].min()
            pcl[:,2] += offset 

        xml_segments = [xml_head]
        for i in range(pcl.shape[0]):
            if precomputed_color is not None:
                color = precomputed_color[i]
            elif use_loc_color and not same_computed_loc_color: 
                color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
            elif use_loc_color and same_computed_loc_color:
                if pcli == 0:
                    color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
                    color_list.append(color)
                else:
                    color = color_list[i] # same color as first shape 
            elif is_color_list:
                color = colorm[pcli]
                color = [c/255.0 for c in color] 
            else:
                color = [c/255.0 for c in colorm] 
            xml_segments.append(xml_ball_segment[material_id].format(
                ball_size, 
                pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        ## print('using color: ', color)
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        if not os.path.exists(os.path.dirname(xmlFile)):
            os.makedirs(os.path.dirname(xmlFile))
        with open(xmlFile, 'w') as f:
            f.write(xml_content)
        # logger.info('[render_mitsuba_pc] write output at: {}', xmlFile)
        f.close()

        if not os.path.exists(os.path.dirname(exrFile)):
            os.makedirs(os.path.dirname(exrFile))
        if not os.path.exists(os.path.dirname(png)):
            os.makedirs(os.path.dirname(png))
        # logger.info('*'*20 + f'{png}' +'*'*20)
        # mitsuba2 
        #subprocess.run([PATH_TO_MITSUBA2, '-o', exrFile, xmlFile])  
        #ConvertEXRToJPG(exrFile, png, trim_img)
        scene = mi.load_file(xmlFile) 
        image = mi.render(scene) ##, spp=256) 
        mi.util.write_bitmap(png, image) 
        if trim_img:
            img = Image.open(png) 
            img.save(png) 
    return png 