import xlrd
import numpy as np
import matplotlib.pyplot as plt


def input():
    loaction = r'F:/综合能源/数据/住宅数据.xls'
    workbook = xlrd.open_workbook(loaction)
    sheet = workbook.sheet_by_name('Load')

    data = np.mat(np.zeros((8760, 13)))
    for i in range(1, 8761):
        data[i-1, :] = sheet.row_values(i)[0:13]

    data = np.array(data).astype('float64')

    return data


def main():
    origin_data = input()
    print(origin_data.shape)
    data = origin_data[:, [3, 4, 6]]
    elec = data[0:30, 2]
    # 加入噪声
    normal_factor = 0.5
    elec_noisy = elec + normal_factor * np.random.normal(loc=0, scale=1, size=elec.shape)
    plt.subplot(3, 1, 1)
    plt.plot(range(30), data[0:30, 0], label='cool', color='b')
    plt.subplot(3, 1, 2)
    # plt.plot(range(30), data[0:30, 1], label='steam', color='r')
    plt.plot(range(30), elec_noisy, label='elec_noisy', color='r')
    plt.subplot(3, 1, 3)
    plt.plot(range(30), data[0:30, 2], label='elec', color='g')
    plt.legend(['cool', 'steam', 'elec'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
