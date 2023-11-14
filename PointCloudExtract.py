from OutPut import *

class PointCloud:
    BinaryImg = None
    Nodeset = None
    Boudnum = None
    Elements = None
    Uppernum = None

    def __init__(self, BinaryImg):
        self.BinaryImg = BinaryImg


    def setNodeset(self):
        self.Nodeset = np.array(np.nonzero(self.BinaryImg)).transpose()


    def setBoudnum(self):
        Boudnum = []
        cells = np.nonzero(self.BinaryImg)
        cellsx = cells[0]
        cellsy = cells[1]
        Neighbors = [(1, 1), (1, -1), (1, 0), (-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)]
        for i in range(cellsx.shape[0]):
            sum_neighbors = 0
            for neighbor in Neighbors:
                dr, dc = neighbor
                sum_neighbors += self.BinaryImg[cellsx[i] + dr][cellsy[i] + dc]
            if (sum_neighbors < 8).all():
                Boudnum.append(i)

        self.Boudnum = np.array(Boudnum)




    def setElements(self):
        cellzrange = np.arange(self.Nodeset[:, 2].min(), 1, self.Nodeset[:, 2].max())
        Nodeset = self.Nodeset
        LabelBinaryImg = -np.ones((self.BinaryImg.shape))
        for i in range(Nodeset.shape[0]):
            LabelBinaryImg[Nodeset[i, 0], Nodeset[i, 1], Nodeset[i, 2]] = i

        #Nodeset = np.hstack((np.arange(0, Nodeset.shape[0])[:, np.newaxis], Nodeset))
        Elements = np.zeros((Nodeset.shape[0], 8), dtype=int)
        BinaryImg = self.BinaryImg
        k = 0
        for i in range(Nodeset.shape[0]):
            p = Nodeset[i, :]
            s = BinaryImg[p[0] + 1, p[1], p[2]] + BinaryImg[p[0], p[1] + 1, p[2]] + BinaryImg[
                p[0] + 1, p[1] + 1, p[2]] \
                + BinaryImg[p[0], p[1], p[2] + 1] + BinaryImg[p[0] + 1, p[1], p[2] + 1] + BinaryImg[
                    p[0], p[1] + 1, p[2] + 1] \
                + BinaryImg[p[0] + 1, p[1] + 1, p[2] + 1]
            if s == 7:  # a cube consisted by this point and its 7 neighbors at right upper
                Idnode = [LabelBinaryImg[p[0], p[1], p[2]], LabelBinaryImg[p[0] + 1, p[1], p[2]]
                    , LabelBinaryImg[p[0], p[1] + 1, p[2]], LabelBinaryImg[p[0] + 1, p[1] + 1, p[2]]
                    , LabelBinaryImg[p[0], p[1], p[2] + 1], LabelBinaryImg[p[0] + 1, p[1], p[2] + 1]
                    , LabelBinaryImg[p[0], p[1] + 1, p[2] + 1], LabelBinaryImg[p[0] + 1, p[1] + 1, p[2] + 1]]
                Elements[k, :] = Idnode
                k = k + 1
        self.Elements = Elements[0:k, :]

    def setUppernum(self):
        Nodes_z = self.Nodeset[:, 2]
        self.Uppernum = np.squeeze(np.where(Nodes_z == Nodes_z.max()))

    def write_output(self, plyname, bouname, uppername):
        self.Elements, Faces = hex2tri(self.Elements)
        print('hex elements are converted to tri elements')
        PointCloud2Ply(self.Nodeset, Faces, plyname)
        PointCloud2bou(self.Boudnum, bouname)
        PointCloud2bou(self.Uppernum, uppername)

    def getdisp(self, objname, m, n):
        with open(r'/home/fty/elastic_mat/cmake_copy_SANM/SANM/config/celltest1-i0-neohookean_i.obj') as file:
            data = file.readlines()
            Points = []
            for i in range(len(data)):
                linelist = data[i][:-1].split(' ')
                if linelist[0] != 'v':
                    break
                else:
                    Points.append(linelist[1:])

            Points = 12.5 * np.array(np.double(Points))
            file.close()

        ####post prossesing
        # calculate the disp
        disp = Points[:, 2] - self.Nodeset[:, 2]
        Points = np.int16(np.around(Points))
        dispmap = np.zeros([m, n])
        effline = np.squeeze(np.array(np.nonzero(disp)))
        for i in range(effline.__len__()):
            dispmap[Points[effline[i], 0], Points[effline[i], 1]] = max(
                dispmap[Points[effline[i], 0], Points[effline[i], 1]], abs(disp[effline[i]]))

        return dispmap