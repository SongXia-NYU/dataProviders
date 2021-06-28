from GaussUtils.GaussInfo import read_gauss_log


if __name__ == '__main__':
    read_gauss_log("/ext3/lipop_logs/*.log", ".")
