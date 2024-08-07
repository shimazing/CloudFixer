import os
import glob
import random
from PIL import Image, ImageChops

import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")


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


def standardize_bbox(pcl, return_center_scale=0):
    if torch.is_tensor(pcl):
        pcl = pcl.numpy()
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
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


def trim(im):
     bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
     diff = ImageChops.difference(im, bg)
     bbox = diff.getbbox()
     if bbox: 
         return im.crop(bbox)
     else:
         return im


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def visualize_pclist(input_pts, file_name, colorm=[24,107,239], 
        skip_if_exists=False, is_color_list=False, 
        sample_count=256, out_width=1200, out_height=1200, 
        ball_size=0.025, do_standardize=1, same_computed_loc_color=0, material_id=0, precomputed_color=None,
        use_loc_color=False, lookat_1=3, lookat_2=3, lookat_3=3, do_transform=0):
    """
    Argus: 
        input_pts: (B,N,3) the points to be render 
        file_name: list; output image name 
    """
    assert(len(input_pts.shape) == 3), f'expect: B,N,3; get: {input_pts.shape}'
    assert(type(file_name) is list), f'require file_name as list'

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
    xml_head = xml_head_segment.format(lookat_1, lookat_2, lookat_3, sample_count, out_width, out_height)
    input_pts = input_pts.cpu()
    color_list = []
    for pcli in range(0, input_pts.shape[0]):
        png = file_name[pcli]
        if skip_if_exists and os.path.exists(png):
            continue

        # preprocessing
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

        # create xml
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
        xml_segments.append(xml_tail)
        xml_content = str.join('', xml_segments)

        if not os.path.exists(os.path.dirname(png)):
            os.makedirs(os.path.dirname(png))

        scene = mi.load_string(xml_content)
        image = mi.render(scene)
        mi.util.write_bitmap(png, image)

    for pcli in range(0, input_pts.shape[0]):
        png = file_name[pcli]
        fsize = os.path.getsize(png)
        if fsize > 0:
            img = Image.open(png)
            img = trim(img)
            img = expand2square(img, (255, 255, 255))
            img = img.resize((600, 600))
            img.save(png)