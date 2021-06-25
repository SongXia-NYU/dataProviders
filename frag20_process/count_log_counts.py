from glob import glob
import os
import os.path as osp

if __name__ == '__main__':
    total_count = 0
    for n_heavy in range(9, 21):
        if n_heavy >= 10:
            f_dir = f"/ext3/Frag20_{n_heavy}/gas_opt_log"
            files = glob(osp.join(f_dir, "*.opt.log"))
            print(f"N_logs heavy: {n_heavy}, num: {len(files)}")
            total_count += len(files)
        else:
            for src in ["pubchem", "zinc"]:
                f_dir = f"/ext3/Frag20_{n_heavy}/{src}_gas_opt_log"
                files = glob(osp.join(f_dir, "*.opt.log"))
                print(f"N_logs heavy: {src}_{n_heavy}, num: {len(files)}")
                total_count += len(files)
    print(f"{total_count = }")
