from GaussUtils.GaussInfo import sdf_to_pt_custom

if __name__ == '__main__':
    sdf_to_pt_custom("../sol_data/BBBP.csv", None, "../sol_data/raw/BBBP_mmff_sdfs", ".", "BBBP", "mmff")
