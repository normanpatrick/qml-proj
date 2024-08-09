import misc
import argparse

def main(args):
    topdir_dataset = "~/1/dataset/al5083"
    topdir_preprocess = "~/1/dataset/preprocess"
    json_files = [
        ("train/train.json", "train"),
        ("test/test.json", "test")
]
    rsize = (args.row, args.col)
    normalize = not args.skip_normalize

    print("Parameters")
    print(f"  + resize   : {rsize}")
    print(f"  + normalize: {normalize}")
    if not args.real_run:
        print("Dry-run - exiting. Use --real-run")
        return

    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    da.setup(max_items_per_label=0, resize=rsize, normalize=normalize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=60)
    parser.add_argument("--col", type=int, default=50)
    parser.add_argument("--skip-normalize", action='store_true')
    parser.add_argument("--real-run", action='store_true')
    args = parser.parse_args()
    main(args)
