from glob import glob
import os
import os.path as osp

if __name__ == '__main__':
    for phase in ["gas", "water", "oct"]:
        folder_name = f"raw/plati20_{phase}_coms"
        com_list = [osp.basename(i).split(".")[0] for i in glob(osp.join(folder_name, "*.com"))]
        sdf_list = [osp.basename(i).split(".")[0] for i in glob(osp.join(folder_name, "*.sdf"))]
        failed_set = set(com_list).difference(set(sdf_list))
        with open("plati20_round2.txt", "a") as f:
            for i in failed_set:
                f.write(f"{folder_name}/{i}.com")
