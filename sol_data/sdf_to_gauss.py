import os
import os.path as osp
from glob import glob


def main(input_dir, output_dir, header="header.txt"):
    # *.sdf -> *.com #
    # obConversion = openbabel.OBConversion()
    # obConversion.SetInAndOutFormats("sdf", "com")
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(osp.join(input_dir, "*.sdf"))

    for input_file in input_files:
        print(input_file)
        basename = osp.basename(input_file)
        output_file = osp.join(output_dir, basename.split(".")[0]+".com")
        tmp_file = osp.join(output_dir, basename.split(".")[0]+".tmp")
        os.system("obabel -isdf {} -ocom -O {}".format(input_file, output_file))

        cmd1 = "tail -n+5 {} > {}".format(output_file, tmp_file)
        cmd2 = "cat {} {} > {}".format(header, tmp_file, output_file)
        os.system(cmd1)
        os.system(cmd2)
        os.system("rm {}".format(tmp_file))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", type=str)
    # parser.add_argument("--output_dir", type=str)
    # parser.add_argument("--header", type=str, default="header.txt")
    # args = parser.parse_args()
    # main(args.input_dir, args.output_dir, args.header)
    main("raw/csd20_sdfs", "raw/csd20_water_coms", header="header-water.txt")
    main("raw/csd20_sdfs", "raw/csd20_oct_coms", header="header-oct.txt")
    main("raw/csd20_sdfs", "raw/csd20_gas_coms", header="header-gas.txt")

