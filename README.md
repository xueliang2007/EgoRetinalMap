## EgoRetinalMap
- Ref: [2016-CVPR] Park Egocentric Future Localization.pdf

## Dataset
- Simulate data, which cam's pitch = 20

## Coordonate
- VehCoordinateFrame: X->left, Y->up, Z->front  
- CamCoordinateFrame: X->left, Y->down, Z->front
 
## Output
- output_path: "cfgs['root']/../all_case_results"
- output_list:
```
    -root
        |- case1
        |   |- images
                |-xxx.disp_ego.tiff
                |-xxx.mask_ego.png
                |-xxx.traj_ego.png
        |   |- traj_jsons
                |-xxx.ie.json or xxx.json
        |- case2
        |- ...

    - 'xxx.ie.json' json_format
        - traj_iuvs: uv of traj on ego_image
        - traj_euvs: uv of traj on ego_retinal_image
        - img_hw
        - cam_k
    - 'xxx.json' json_format
        - traj_iuvs: uv of traj on ego_image
``` 

