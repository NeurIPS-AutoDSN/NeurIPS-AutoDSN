import numpy as np

genotype_path = r'./genotype.npy'


def select_genotype(genotype_path, high_order, topk=7, topks=''):

    if high_order:
        if topks:
            topk_list = list(map(int, topks.split('_')))
        else:
            topk_list = [topk, topk, 5]
        print(topk_list)
        genotype = np.load(genotype_path, allow_pickle=True).item()
        print("origin 2nd:", genotype['2nd'][:, 0])
        print("origin 3rd:", genotype['3rd'][:, 0])
        genotype['3rd'] = genotype['3rd'] / genotype['3rd'].sum()

        index_2nd = np.argsort(genotype['2nd'], axis=0)
        index_3rd = np.argsort(genotype['3rd'], axis=0)
        thr_2nd = genotype['2nd'][index_2nd[-topk_list[0]]].item()
        thr_3rd = genotype['3rd'][index_3rd[-topk_list[1]]].item()
        genotype['2nd'][genotype['2nd'] < thr_2nd] = 0
        genotype['3rd'][genotype['3rd'] < thr_3rd] = 0
        print("selected 2nd:", genotype['2nd'][:, 0])
        print("selected 3rd:", genotype['3rd'][:, 0])
    else:
        genotype = np.load(genotype_path)
        print("origin:", genotype[:, 0])
        index = np.argsort(genotype, axis=0)
        print("index", index[::-1])
        thr = genotype[index[-topk]].item()

        genotype[genotype < thr] = 0
        print("selected:", genotype[:, 0])

    return genotype


if __name__ == "__main__":
    genotype = select_genotype(genotype_path, high_order=True)
