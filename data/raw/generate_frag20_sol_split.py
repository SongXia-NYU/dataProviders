import numpy as np
import torch

if __name__ == '__main__':
    prev_n = 0
    train_index = []
    test_index = []
    for n_heavy in range(9, 21):
        f_name = "fragment{}_split.npz".format(n_heavy)
        data = np.load(f_name)
        this_train = data["train"]
        this_test = data["test"]

        train_index.append(torch.as_tensor(this_train) + prev_n)
        test_index.append(torch.as_tensor(this_test) + prev_n)
        prev_n += len(this_train) + len(this_test)
    train_index = torch.cat(train_index)
    test_index = torch.cat(test_index)

    j2d_map = torch.load("jianing_to_dongdong_merge.pt")
    j2d_index_map = torch.where(j2d_map == 0, -1, j2d_map)
    count = 0
    for i in range(len(j2d_index_map)):
        if j2d_index_map[i] > 0:
            j2d_index_map[i] = count
            count += 1
    _tmp_train = set(j2d_index_map[train_index].tolist())
    _tmp_train.remove(-1)
    train_index_dd = torch.as_tensor(list(_tmp_train))
    train_index_dd = train_index_dd[torch.randperm(len(train_index_dd))]
    _tmp_test = set(j2d_index_map[test_index].tolist())
    _tmp_test.remove(-1)
    test_index_dd = torch.as_tensor(list(_tmp_test))
    print("-1 count: ", (train_index_dd == -1).sum())
    print("-1 count: ", (test_index_dd == -1).sum())
    torch.save({"train_index": train_index_dd[:-1000],
                "val_index": train_index_dd[-1000:],
                "test_index": test_index_dd},
               "../processed/frag20_sol_split.pt")
    print("finished")
