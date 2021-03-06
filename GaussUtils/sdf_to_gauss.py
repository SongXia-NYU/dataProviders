from openbabel import openbabel
import argparse
from glob import glob
import os.path as osp
import os


def main(input_dir, output_dir, header):
    # *.sdf -> *.com #
    # obConversion = openbabel.OBConversion()
    # obConversion.SetInAndOutFormats("sdf", "com")
    input_files = glob(osp.join(input_dir, "*.sdf"))

    for input_file in input_files:
        basename = osp.basename(input_file)
        output_file = osp.join(output_dir, basename.split(".")[0]+".com")
        tmp_file = osp.join(output_dir, basename.split(".")[0]+".tmp")
        os.system("obabel -isdf {} -ocom -O {}".format(input_file, output_file))

        cmd1 = "tail -n+4 {} > {}".format(output_file, tmp_file)
        cmd2 = "cat {} {} > {}".format(header, tmp_file, output_file)
        os.system(cmd1)
        os.system(cmd2)
        os.system("rm {}".format(tmp_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--header", type=str, default="header.txt")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.header)
