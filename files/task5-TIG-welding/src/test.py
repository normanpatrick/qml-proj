import os
import numpy as np
import misc
from sklearn.model_selection import train_test_split

topdir_dataset = "~/1/dataset/al5083"
topdir_preprocess = "~/1/dataset/preprocess"
json_files = [
    ("train/train.json", "train"),
    ("test/test.json", "test")
]

def _helper_01(rsize=None, details=True, normalize=True):
    td = misc.TIGDataset(topdir=topdir_dataset,
                         resize=rsize,
                         normalize=normalize,
                         how_many_classes=6,
                         json_files=json_files)
    if details:
        print(td)
    samples = td.samples({0:10, 1:11, 5:15})

    for k in samples:
        fst = samples[k][0]
        print(f"label {k}: {len(samples[k])} images")
        print("Shape:", fst.shape)
        print("Sum:", np.sum(fst))
        print("Max:", np.max(fst),
              ", Min:", np.min(fst),
              ", Mean:", np.sum(fst)/fst.size)

def test_01():
    for nm in [False, True]:
        print(f"---------------------- normalize = {nm} ---------------------")
        print("==== No compression ====")
        _helper_01(normalize=nm)
        print("==== With compression ====")
        _helper_01((60,50), details=False, normalize=nm)

def test_02():
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    da.setup(max_items_per_label=4, resize=rsize, normalize=normalize)

def test_03():
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    da.setup(max_items_per_label=10,
             force=True,
             resize=rsize,
             normalize=normalize)

def test_04():
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    X, y = da.load_data(max_items_per_label=4,
                        resize=rsize,
                        normalize=normalize)
    seed = 101
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      random_state=seed,
                                                      test_size=0.3)
    print("Original :", X.shape, y.shape)
    print("training :", X_train.shape, y_train.shape)
    print("validation:", X_val.shape, y_val.shape)

def test_05():
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    X, y = da.load_data(max_items_per_label=2000,
                        resize=rsize,
                        normalize=normalize)
    print("Original :", X.shape, y.shape)
    print(y[:16])

def test_06():
    seed = 101
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    X, y = da.load_data(max_items_per_label=100,
                        resize=rsize,
                        normalize=normalize)
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      random_state=seed,
                                                      test_size=0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                    y_val,
                                                    random_state=seed,
                                                    test_size=0.2)

    print("X, y", X.shape, y.shape)
    print("X_train, y_train",
          X_train.shape, y_train.shape)
    print("X_val, X_test, y_val, y_test",
          X_val.shape, X_test.shape, y_val.shape, y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test 
    
def main():
    test_fns = [
        lambda: print("This is a null test"),
        test_01,
        test_02,
        test_03,
        test_04,
        test_05,
        test_06,
    ]
    max_tid = len(test_fns) - 1

    tid = os.environ.get("TIGTEST")
    if tid:
        tid = int(tid)
    else:
        tid = 0
        print(f"Set env var TIGTEST to 1...{max_tid} for a test")

    if tid > max_tid:
        print("Valid tests ids are 1...{max_tid}")
        return
    test_fns[tid]()

if __name__ == '__main__':
    main()
