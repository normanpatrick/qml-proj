"""
Author: Nirmalendu B Patra
Copyright (C) Nirmalendu B Patra - All Rights Reserved
"""

import os
import json
import random
import hashlib
import numpy as np
from pprint import pprint
from PIL import Image
import skimage.transform

class TIGDataAccess(object):
    """
    - Pre-process, save normalized, non-normalized all in npy format
    - Should behave like a seamless cache
    """
    def __init__(self,
                 dir_in,
                 json_files,
                 how_many_classes,
                 dir_preprocess):
        self.dir_in = dir_in
        self.dir_preprocess = os.path.expanduser(dir_preprocess)
        self.json_files = json_files
        self.how_many_classes=how_many_classes
        self.td = None
        self.uniq_name = \
            lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()


    def _fname_meta(self, norm, resize):
        row, col = resize
        n_pth = "norm" if norm else "not-normed"
        return os.path.join(self.dir_preprocess,
                            f"dataset_{n_pth}_r{row}_c{col}.json")

    # we need a way to create unique filepath since the filenames
    # in the TIG dataset are not unique at all
    def _img_path(self, norm, label, resize, fname):
        fname_npy = os.path.splitext(fname)[0] + ".npy"
        row, col = resize
        n_pth = "normalized" if norm else "not-normalized"
        l = f"label-{label}"
        pdir = f"{n_pth}/{l}/r{row}/c{col}"
        pth = f"{pdir}/{fname_npy}"
        return (
            os.path.join(self.dir_preprocess, pth),
            os.path.join(self.dir_preprocess, pdir)
        )

    def load_data(self,
                  resize,
                  normalize,
                  force=False,
                  max_items_per_label=0):
        data_X = []
        data_y = []
        f_meta = self._fname_meta(norm=normalize, resize=resize)
        assert os.path.exists(f_meta), "dataset meta file must exist"
        with open(f_meta) as f:
            jdata = json.loads(f.read())
            # pprint(jdata["npy"])
            labels = jdata["meta"].keys()
            for l in labels:
                for ii in range(max_items_per_label):
                    npy_fname = jdata["npy"][l][ii]
                    data_X.append(np.load(npy_fname))
                    # labels/class are integers
                    data_y.append(int(l))
        return np.array(data_X), np.array(data_y)

    """
    - pre-process all images
    - save normalized, non-normalized all in npy format
    - allow for various resolutions
    - API should support sklearn.train_test_split
    """
    def setup(self,
              resize,
              normalize,
              force=False,
              max_items_per_label=0):
        if self.td is None:
            metadata = {k:0 for k in range(self.how_many_classes)}
            flist_npy = {k:[] for k in range(self.how_many_classes)}
            fset = set()
            os.makedirs(self.dir_preprocess, exist_ok=True)
            self.td = TIGDataset(topdir=self.dir_in,
                                 json_files=self.json_files,
                                 how_many_classes=self.how_many_classes,
                                 resize=resize,
                                 normalize=normalize,
                                 max_items_per_label=max_items_per_label)
            total = 0
            for label in range(self.how_many_classes):
                for img in self.td.next_image(label=label):
                    total += 1
                    fname =  self.uniq_name(img) # os.path.split(img)[-1]
                    i_p, i_dir = self._img_path(norm=normalize,
                                                label=label,
                                                resize=resize,
                                                fname=fname)
                    assert i_p not in fset, "there should be no duplicates"
                    fset.add(i_p)
                    os.makedirs(i_dir, exist_ok=True)
                    metadata[label] += 1
                    flist_npy[label].append(i_p)
                    if force or (not os.path.exists(i_p)):
                        imdata = self.td.process_single_image(img)
                        np.save(i_p, imdata)
                    # print("wahoo", i_p, os.path.exists(i_p))
            print(f"Total processed: {total}")
            print(metadata)
            f_meta = self._fname_meta(norm=normalize, resize=resize)
            if force or (not os.path.exists(f_meta)):
                with open(f_meta, "w") as f:
                    f.write(json.dumps({
                        "meta": metadata,
                        "npy": flist_npy,
                        },
                        indent=4))

class TIGDataset(object):
    def __init__(self,
                 topdir,
                 json_files,
                 how_many_classes,
                 max_items_per_label=0,
                 resize=None,
                 normalize=True):
        self.topdir = os.path.expanduser(topdir)
        self.json_files = json_files
        self.how_many_classes = how_many_classes
        self.dataset = {k: [] for k in range(how_many_classes)}
        self.duplicates = []
        self.resize = resize
        self.normalize = normalize
        # if n > 0, then process only n items, otherwise all
        self.max_items_per_label = max_items_per_label
        self._consolidate()
        # random.seed(seed)
        # for label in self.dataset:
        #     random.shuffle(self.dataset[label])

    def _consolidate(self):
        already_seen = set()
        totals = {k:0 for k in range(self.how_many_classes)}
        for jf, jdir in self.json_files:
            # print(os.path.join(self.topdir, jf))
            with open(os.path.join(self.topdir, jf)) as f:
                jdata = json.loads(f.read())
                for img,label in jdata.items():
                    if self.max_items_per_label == totals[label] \
                       and self.max_items_per_label > 0:
                        continue
                    img_path = os.path.join(self.topdir, jdir, img)
                    if img_path not in already_seen:
                        totals[label] += 1
                        already_seen.add(img_path)
                        self.dataset[label].append(img_path)
                    else:
                        self.duplicates.append(img_path)

    def next_image(self, label):
        for item in self.dataset[label]:
            yield item

    def process_single_image(self, img):
        assert os.path.exists(img) == True
        imdata = self._image(img)
        if self.normalize:
            imdata = self._image(img) / 255 # np.linalg.norm(imdata)
        return imdata

    def samples(self, d_how_many):
        res = {}
        for label, num in d_how_many.items():
            res[label] = []
            for img in self.dataset[label][:num]:
                res[label].append(self.process_single_image(img))
        return res

    def _image(self, fname):
        img = Image.open(fname)
        img.load()
        npi = np.asarray(img, dtype="int32")
        return npi if self.resize is None \
            else skimage.transform.resize(npi, self.resize, preserve_range=True)

    def __repr__(self):
        dup = len(self.duplicates)
        total = 0
        rs = "no" if self.resize is None else self.resize
        s = f"TIGDataset (resize: {rs})\n"
        for k in self.dataset.keys():
            samples = len(self.dataset[k])
            total += samples
            s += f"  + label {k}: {samples}\n"
        s +=f"  - {dup} duplicates found, total {total} images"
        return s

if __name__ == '__main__':
    pass
