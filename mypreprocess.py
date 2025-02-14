from dataclasses import dataclass, field
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import random
import pickle
import argparse
from typing import Any, Dict, List, Sequence, Tuple, Type
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from tqdm import tqdm
# from eval.openclip_encoder import OpenCLIPNetwork
from itertools import groupby
import cProfile
from transformers import CLIPModel, CLIPProcessor

image_path = None
save_path = None
device = torch.device('cuda:0')
tb_writer = None

class Segments:
    smaps: List[np.ndarray]
    def __init__(self, image_num, image_height, image_width) -> None:
        self.image_num = image_num
        self.image_height = image_height
        self.image_width = image_width
        self.cursor = 0
        self.smaps = [np.full((image_height, image_width), -1, dtype=np.int32) for i in range(image_num)]
        
    def remove_duplicate(self, frame_idx, object_ids, masks, prompt: list):
        smap = self.smaps[frame_idx]
        ret = []
        for i, mask in enumerate(masks):
            if duplicate(smap, mask)<0.8:
                ret.append(prompt[i])
        # print(f'remove {len(prompt)-len(ret)} at {frame_idx}')
        return ret
    
    def add_masks(self, frame_idx, object_ids, masks):
        smap=self.smaps[frame_idx]
        for id, mask in zip(object_ids, masks, strict=True):
            smap[mask] = id

class Entities:
    entities: list
    def __init__(self, image_num, image_height, image_width) -> None:
        self.entities = []
        self.image_num=image_num
        self.image_height=image_height
        self.image_width=image_width

    def __getitem__(self, index):
        return self.entities[index]

    def add_entities(self, current_frame, ids: list, masks, prompt):
        object_ids = []
        for i, mask in zip(ids, masks, strict=True):
            object_ids.append(len(self.entities))
            self.entities.append({
                'prompt_frame': [current_frame],
                # 'prompt': prompt[i],
                # 'mask': mask,
                'rles': [None for i in range(self.image_num)]
            })
        # print(f'add {len(object_ids)} at {current_frame} total {len(self.entities)}')
        return object_ids
    
    def add_entity(self, prompt_frame, rles):
        self.entities.append({
                'prompt_frame': [prompt_frame],
                'rles': rles
            })

    def add_masks(self, frame_idx, entity_ids, masks):
        for id, mask in zip(entity_ids, masks, strict=True):
            self.entities[id]['rles'][frame_idx]=mask_to_rle(mask)

    def remove_duplicate(self, frame_idx, entity_ids, masks_entity, prompt: list):
        ret = []
        for i, mask in enumerate(masks_entity):
            if self.duplicate_id(frame_idx, mask)==None and mask.sum()!=0:
                ret.append(prompt[i])
        # print(f'remove {len(prompt)-len(ret)} at {frame_idx}')
        return ret
    
    def duplicate_id(self, frame_idx, mask)->int|None:
        if len(self.entities)==0:
            return None
        masks_frame = self.get_masks_by_frame(frame_idx)
        ious: np.ndarray = (mask&masks_frame).sum((1,2))/(mask|masks_frame).sum((1,2)) # (entities)
        ious = np.nan_to_num(ious, nan=0, posinf=1, neginf=0)
        idx=np.argmax(ious)
        if ious[idx]>0.8:
            return idx
        else:
            return None

    def remove(self, id):
        self.entities.pop(id)

    def get_masks_by_frame(self, frame_idx)->np.ndarray:
        if len(self.entities)==0:
            return None
        ret=[]
        for entity in self.entities:
            ret.append(rle_to_mask(entity['rles'][frame_idx]))
        return np.stack(ret)

    def get_masks_by_entity(self, entity_id)->np.ndarray:
        ret=[]
        for rle in self.entities[entity_id]['rles']:
            ret.append(rle_to_mask(rle))
        return np.stack(ret)

    def get_colormap(self):
        colormap=[torch.rand(3) for i in range(len(self.entities))]
        colormap.append(torch.zeros(3))
        return torch.stack(colormap).cuda()
    
    def get_segments(self)->np.ndarray:
        ret=np.full((self.image_num, self.image_height, self.image_width), -1)
        for frame_idx in tqdm(list(range(self.image_num)), desc='cal segments'):
            masks = self.get_masks_by_frame(frame_idx)
            masks = sorted(enumerate(masks), key=lambda mask: mask[1].sum(), reverse=True)
            for id, mask in masks:
                ret[frame_idx, mask] = id
        return ret
    
def duplicate(smap, mask):
    smap=smap>=0
    mask=mask>0
    return (smap&mask).sum()/mask.sum()

def calculate_iou(mask1, mask2):
    # 计算两个 mask 的交集和并集
    mask1 = mask1>0
    mask2 = mask2>0
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    return intersection / union if union > 0 else 0

def save_smap(segments: Segments, entities: Entities):
    colormap = entities.get_colormap()
    for i, smap in enumerate(segments.smaps):
        fmap=colormap[smap]
        torchvision.utils.save_image(fmap.permute(2,0,1), os.path.join(save_path, f'{str(i).rjust(5,'0')}.jpg'))

def get_entities(predictor: SAM2VideoPredictor, state, frame_idx, prompt):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # state = predictor.init_state(image_path)
        predictor.reset_state(state)
        for id, p in enumerate(prompt):
            predictor.add_new_mask(state, frame_idx, id, p)
        for frame_index, object_ids, masks in predictor.propagate_in_video(state): # masks: (n, 1, h, w)
            masks = (masks.squeeze(1)>0).clone().detach().cpu().numpy()
            if frame_index == frame_idx:
                break
    return frame_index, object_ids.copy(), masks

def mask_to_rle(mask: np.ndarray) -> List[Dict[str, Any]]|Dict[str, Any]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    if len(mask.shape)==2:
        masks=mask[None]
    else:
        masks=mask
    # Put in fortran order and flatten h,w
    b, h, w = masks.shape
    masks = masks.transpose(0, 2, 1).reshape(b,-1)

    # Compute change indices
    diff = masks[:, 1:] ^ masks[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    rles = []
    for i in range(b):
        cur_idxs = change_indices[1][change_indices[0] == i]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if masks[i, 0] == 0 else [0]
        counts.extend(btw_idxs.tolist())
        rles.append({"size": [h, w], "counts": counts})
    if len(mask.shape)==2:
        return rles[0]
    return rles

def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

def prompt_filter(mask: np.ndarray):
    mask=mask['segmentation']
    area = mask.sum()
    x = np.zeros_like(mask)
    x[0, ...] = True
    x[-1, ...] = True
    x[..., 0] = True
    x[..., -1] = True
    return (~np.any(mask & x)) or (area/mask.size>0.01)

def get_prompt(mask_generator: SAM2AutomaticMaskGenerator, image: np.ndarray):
    records=mask_generator.generate(image) # 2.5G
    records = remove_duplicate_prompt(records)
    return [record['segmentation'] for record in records if prompt_filter(record)]

def combine_records(record1, record2):
    return {
        'segmentation': record1['segmentation']|record2['segmentation']
    }

def remove_duplicate_prompt(records:list):
    # 存储有效的 mask
    unique_records = []
    
    for i, record in enumerate(records):
        # 检查当前 mask 是否与 unique_masks 中的任何 mask 重叠
        is_duplicate = False
        for idx, unique_record in enumerate(unique_records):
            if calculate_iou(record['segmentation'], unique_record['segmentation']) > 0.8:
                is_duplicate = True
                unique_records[idx] = combine_records(record, unique_record)
        
        if not is_duplicate:
            unique_records.append(record)
    
    return unique_records


def prompt_filter_bbox(record):
    bbox=record['bbox']
    x,y,w,h = bbox
    image_width, image_height = record['segmentation'].shape
    if x!=0 and y!=0 and x+w!=image_width and y+h!=image_height:
        return True
    return False

def get_prompt_bbox(mask_generator, image: np.ndarray):
    records=mask_generator.generate(image)
    ret= [record['bbox'] for record in records if prompt_filter_bbox(record)]
    return ret

def mask_or(*masks):
    masks = [m>0 for m in masks]
    ret=torch.zeros_like(masks[0])
    for m in masks:
        ret=ret|m
    return ret

def video_segment(image_names: List[str], images: np.ndarray):
    global image_path, save_path, args
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(args.sam_path, 
                                                                points_per_side=32,
                                                                pred_iou_thresh=0.7,
                                                                box_nms_thresh=0.7,
                                                                stability_score_thresh=0.85,
                                                                # crop_n_layers=1,
                                                                # crop_n_points_downscale_factor=1,
                                                                min_mask_region_area=100,
                                                                # output_mode='uncompressed_rle'
                                                                ) # 1G
    predictor = SAM2VideoPredictor.from_pretrained(args.sam_path) #1G
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(image_path) # 3G

    entities = Entities(len(images), images.shape[1], images.shape[2])
    for current_frame, image_name, image in tqdm(list(zip(range(len(images)), image_names, images, strict=True)), desc='video_segment'):
        prompt = get_prompt(mask_generator, image)
        if len(prompt)==0:
            continue
        frame_idx, entity_ids, masks = get_entities(predictor, state, current_frame, prompt)
        prompt = entities.remove_duplicate(frame_idx, entity_ids, masks, prompt)
        tqdm.write(f'remove {len(masks)-len(prompt)} prompts')
        if len(prompt)==0:
            continue
        frame_idx, entity_ids, masks = get_entities(predictor, state, current_frame, prompt)
        rles = [[None for i in range(len(images))] for i in range(len(entity_ids))]
        tqdm.write(f'track {len(entity_ids)} entities at {current_frame} total {len(entities.entities)}')

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # state = predictor.init_state(image_path)
            predictor.reset_state(state)
            for id, p in zip(range(len(prompt)), prompt, strict=True):
                predictor.add_new_mask(state, current_frame, id, p)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                masks = (masks.squeeze(1)>0).clone().detach().cpu().numpy()
                for id,rle in zip(object_ids,mask_to_rle(masks), strict=True):
                    rles[id][frame_idx]=rle
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state, reverse=True):
                masks = (masks.squeeze(1)>0).clone().detach().cpu().numpy()
                if frame_idx == current_frame:
                    continue
                for id,rle in zip(object_ids,mask_to_rle(masks), strict=True):
                    rles[id][frame_idx]=rle
        for id in entity_ids:
            entities.add_entity(current_frame, rles[id])
        tqdm.write(f'new {len(entity_ids)} at {current_frame} total {len(entities.entities)}')
    return entities

def get_bbox(mask: np.ndarray):
     # 查找掩码中的 True 元素的索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # 如果没有 True 元素，则返回全零的边界框
    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)
    
    # 获取边界框的上下左右边界
    x_min, x_max = np.where(rows)[0][[0, -1]] # h
    y_min, y_max = np.where(cols)[0][[0, -1]] # w
    
    # 返回边界框
    return (x_min, y_min, x_max + 1 - x_min, y_max + 1 - y_min) # x, y, h, w

def get_entity_image(image: np.ndarray, mask: np.ndarray)->np.ndarray:
    if mask.sum()==0:
        return np.zeros((224,224,3), dtype=np.uint8)
    image = image.copy()
    # crop by bbox
    x,y,h,w = get_bbox(mask)
    image[~mask] = np.zeros(3, dtype=np.uint8) #分割区域外为白色
    image = image[x:x+h, y:y+w, ...] #将img按分割区域bbox裁剪
    # pad to square
    l = max(h,w)
    paded_img = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        paded_img[:,(h-w)//2:(h-w)//2 + w, :] = image
    else:
        paded_img[(w-h)//2:(w-h)//2 + h, :, :] = image
    paded_img = cv2.resize(paded_img, (224,224))
    return paded_img

# def extract_semantics(images: np.ndarray, entities: Entities):
#     global save_path, image_path, args
#     clip = OpenCLIPNetwork()
#     semantics=[]
#     indices=[]
#     for id, entity in tqdm(list(enumerate(entities.entities)), desc='extract semantics'):
#         masks=entities.get_masks_by_entity(id)
#         indice = masks.sum((1,2))!=0
#         entity_images=[]
#         for image,mask in zip(images[indice], masks[indice], strict=True):
#             entity_images.append(get_entity_image(image, mask))
#         entity_images=np.stack(entity_images)
#         with torch.no_grad():
#             semantic0 = clip.encode_image(torch.from_numpy(entity_images).cuda().permute(0, 3, 1, 2)/255)
#             semantic0 = semantic0.clone().detach().cpu().float().numpy()
#             semantic0 /=np.linalg.norm(semantic0, axis=-1, keepdims=True)
#         semantic = np.zeros((len(masks), semantic0.shape[-1]), dtype=semantic0.dtype)
#         semantic[indice]=semantic0
#         # semantic /=semantic.norm(dim=-1, keepdim=True) # (images, 1, 512)
#         semantics.append(semantic)
#         indices.append(indice)
#     semantics = np.stack(semantics)
#     indices=np.stack(indices)
#     # semantics = clip.encode_image(entity_images.permute(0, 3, 1, 2))
#     np.save(os.path.join(save_path, 'raw_semantics.npy'), semantics)
#     np.save(os.path.join(save_path, 'raw_semantics_mask.npy'), indices)

def extract_semantics_hf(images: np.ndarray, entities: Entities):
    global save_path, image_path, args
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    semantics=[]
    indices=[]
    for id, entity in tqdm(list(enumerate(entities.entities)), desc='extract semantics'):
        masks=entities.get_masks_by_entity(id)
        indice = masks.sum((1,2))!=0
        entity_images=[]
        for image,mask in zip(images[indice], masks[indice], strict=True):
            entity_images.append(get_entity_image(image, mask))
        inputs = clip_processor(images=entity_images, return_tensors='pt')
        inputs = inputs.to(clip_model.device)
        semantic0 = clip_model.get_image_features(**inputs)
        semantic0 = F.normalize(semantic0,dim=-1).detach().cpu().numpy()
        semantic = np.zeros((len(masks), semantic0.shape[-1]), dtype=semantic0.dtype)
        semantic[indice]=semantic0
        semantics.append(semantic)
        indices.append(indice)
    semantics = np.stack(semantics)
    indices=np.stack(indices)
    np.save(os.path.join(save_path, 'raw_semantics.npy'), semantics)
    np.save(os.path.join(save_path, 'raw_semantics_mask.npy'), indices)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', '-n', type=str, default='temp')
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--save_folder', type=str, default='semantic')
    parser.add_argument('--sam_path', type=str, default="facebook/sam2-hiera-large")
    parser.add_argument('--flag', action='store_true')
    torch.set_default_dtype(torch.float32)
    return parser.parse_args()

def main() -> None:
    global save_path, image_path, args, tb_writer
    seed_everything(42)
    torch.set_default_device(device)
    args = prepare_args()
    image_path = os.path.join(args.dataset_path, args.image_folder)
    save_path = os.path.join(args.dataset_path, args.save_folder)
    img_list = []
    WARNED = False
    image_names = os.listdir(image_path)
    image_names.sort()
    for image_name in image_names:
        image = cv2.imread(os.path.join(image_path, image_name))

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))

        image = cv2.resize(image, resolution)
        img_list.append(image)
    images = np.stack(img_list)

    os.makedirs(save_path, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    if args.flag:
        with open(os.path.join(save_path, 'entities.pk'), 'rb') as ef:
            entities = pickle.load(ef)
    else:
        entities = video_segment(image_names, images)

    segments = entities.get_segments()
    np.save(os.path.join(save_path, 'segments.npy'), segments)
    for image_name, smap in zip(image_names, segments, strict=True):
        np.save(os.path.join(save_path, f'{os.path.splitext(image_name)[0]}.npy'), smap)
    with open(os.path.join(save_path, 'entities.pk'), 'wb') as ef:
        pickle.dump(entities, ef)

    extract_semantics_hf(images, entities)
    pr.disable()
    pr.dump_stats(os.path.join('profile', f'{os.path.splitext(os.path.basename(args.dataset_name))[0]}.prof'))

if __name__  == '__main__':
    main()