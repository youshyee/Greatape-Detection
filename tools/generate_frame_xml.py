import xml.etree.cElementTree as ET
import os
import mmcv
def create_frame_xml(videoname,videometa,annotations,species,output_path):
    framenum=videometa['frame_ids'][0]+1
    h,w,c=videometa['ori_shape']
    annotations=[a for a in annotations]
    root = ET.Element("annotation")
    ET.SubElement(root, "videoname").text=videoname
    ET.SubElement(root, "frameid").text=str(framenum)
    size=ET.SubElement(root, "size")
    ET.SubElement(size, "height").text=f'{int(h)}'
    ET.SubElement(size, "width").text=f'{int(w)}'
    ET.SubElement(size, "depth").text=f'{int(c)}'

    ET.SubElement(root, "is_object").text='True' if annotations else 'False'
    for annotation in annotations:
        xmin=annotation[0]
        ymin=annotation[1]
        xmax=annotation[2]
        ymax=annotation[3]
        obj=ET.SubElement(root, "object")
        ET.SubElement(obj, "category").text='Great Ape'
        ET.SubElement(obj, "name").text=species
        bnb=ET.SubElement(obj, "bndbox")
        ET.SubElement(bnb, "xmin").text=f'{xmin:.2f}'
        ET.SubElement(bnb, "ymin").text=f'{ymin:.2f}'
        ET.SubElement(bnb, "xmax").text=f'{xmax:.2f}'
        ET.SubElement(bnb, "ymax").text=f'{ymax:.2f}'

    tree = ET.ElementTree(root)
    mmcv.mkdir_or_exist(os.path.join(output_path,f'{videoname}','Annotation'))
    tree.write(os.path.join(output_path,f'{videoname}',f"{videoname}_frame_{framenum}.xml"))