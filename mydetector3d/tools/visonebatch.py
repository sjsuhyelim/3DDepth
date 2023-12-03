import numpy as np
import os
import cv2
import sys
import argparse
import os
# import matplotlib
# matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import pickle 
from visual_utils.mayavivisualize_utils import boxes_to_corners_3d, visualize_pts, draw_lidar, mydraw_scenes, draw_scenes, draw_gt_boxes3d

def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

# pointpillar: /Users/hyelim_yang/Documents/CMPE249/project/waymokitti_models_pointpillar_1201/waymokitti_models_pointpillar_1201_numpy_1.pkl
# pointpillar_resnet: /Users/hyelim_yang/Documents/CMPE249/project/waymokitti_models_pointpillar_resnet_1201/waymokitti_models_pointpillar_resnet_1201_numpy_1.pkl
# centerpoint_pillar: /Users/hyelim_yang/Documents/CMPE249/project/waymokitti_models_centerpoint_pillar_1201/waymokitti_models_centerpoint_pillar_1201_numpy_1.pkl

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchpklfile_path", default='/Users/hyelim_yang/Documents/CMPE249/project/waymokitti_models_centerpoint_pillar_1201/waymokitti_models_centerpoint_pillar_1201_numpy_1.pkl', help="pkl file path"
    )#'./data/waymokittisample'

    parser.add_argument(
        "--dataset", default="waymokitti", help="dataset name" 
    )#waymokitti 

    parser.add_argument(
        "--only_gt", default=False, help="draw only ground truth 3D box"
    )

    args = parser.parse_args()
    import sys
    sys.path.append(r'/Users/hyelim_yang/3DDepth')
    print(args.batchpklfile_path)
    f = open(args.batchpklfile_path, 'rb')   # if only use 'r' for reading; it will show error: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
    save_dict = pickle.load(f)         # load file content as mydict
    f.close()


    idx = save_dict['idx']
    #modelname = save_dict['modelname'] #='myvoxelnext'
    #infer_time = save_dict['infer_time']
    datasetname = save_dict['datasetname'] #='waymokitti'
    batch_dict = save_dict['batch_dict'] #=batch_dict
    if not args.only_gt:
        pred_dicts = save_dict['pred_dicts'] ##batch size array of record_dict{'pred_boxes'[N,7],'pred_scores'[N],'pred_labels'[N]}
        idx_pred_dicts=pred_dicts
        pred_boxes = idx_pred_dicts['pred_boxes'] #[295, 7]
        pred_scores = idx_pred_dicts['pred_scores'] #[295]
        pred_labels = idx_pred_dicts['pred_labels']
        threshold = 0.2
        selectbool = pred_scores > threshold
        pred_boxes = pred_boxes[selectbool,:] #[319, 7]->[58, 7]
        pred_scores = pred_scores[selectbool]
        pred_labels = pred_labels[selectbool]
    annos = save_dict['annos'] #=annos batch_size array, each dict is the Kitti annotation-like format dict (2D box is converted from 3D pred box)

    # #batch_dict data:
        # 'gt_boxes': (16, 16, 8), 16: batch size, 16: number of boxes (many are zeros), 8: boxes value
        # 'points': (302730, 5): 5: add 0 in the left of 4 point features (xyzr)
        # Voxels: (89196, 32, 4) 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
        # Voxel_coords: (89196, 4) (batch_index,z,y,x) added batch_index in dataset.collate_batch
        # Voxel_num_points: (89196,)
    batch_gtboxes=batch_dict['gt_boxes']
    batch_points=batch_dict['points'] #3033181, 6
    batch_voxels=batch_dict['voxels'] #[1502173, 5, 5]
    batch_voxelcoords=batch_dict['voxel_coords']
    batch_voxelnumpoints=batch_dict['voxel_num_points']

    idxinbatch = 1
    selectidx=batch_points[:,0] == idxinbatch # idx in the left of 4 point feature (xyzr)
    idx_points = batch_points[selectidx, 1:5] #N,4 points [191399, 4]
    #print('idxinbatch: ', idxinbatch)
    #print(batch_gtboxes[idxinbatch, :, :].shape)
    idx_gtboxes = batch_gtboxes[idxinbatch, :, :]
    #idx_gtboxes=np.squeeze(batch_gtboxes[idxinbatch, :, :], axis=0) #[104, 8]
    #print(idx_gtboxes.shape)


    

    import mayavi.mlab as mlab
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4] #[-37.44, -74.88, -1, 37.44, 74.88, 2.0] # #[0, -40, -3, 70.4, 40, 1]
    if not isinstance(idx_points, np.ndarray):
        idx_points = idx_points.cpu().numpy()
    draw_lidar(idx_points, fig=fig, pts_scale=5, pc_label=False, color_by_intensity=False, drawregion=True, point_cloud_range=point_cloud_range)
    if idx_gtboxes is not None and not isinstance(idx_gtboxes, np.ndarray):
        idx_gtboxes = idx_gtboxes.cpu().numpy()
    #box3d_pts_3d = compute_box_3d(idx_gtboxes) #3d box coordinate=>get 8 points in camera rect, need 3DObject 
    box3d_pts_3d = boxes_to_corners_3d(idx_gtboxes) #[42,8]->(42, 8, 3)
    #colorlabel=INSTANCE3D_Color[obj.type]
    draw_gt_boxes3d(box3d_pts_3d, fig=fig, color=(1, 1, 1), line_width=1, draw_text=False, label=None) #(n,8,3)

    if args.only_gt:
        mlab.show()

    if not args.only_gt: #pred_boxes is not None and not isinstance(pred_boxes, np.ndarray):
        ref_corners3d = boxes_to_corners_3d(pred_boxes)
        draw_gt_boxes3d(ref_corners3d, fig=fig, color=(0, 1, 0), line_width=1, draw_text=False, label=None) #(n,8,3) # green is pred_boxes
        #fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=None, max_num=300)
        mlab.show()

    #mydraw_scenes(idx_points, idx_gtboxes, pred_boxes)

    print('done')

    #draw_scenes

    


