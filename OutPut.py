import numpy as np

def hex2tri(hexElements):
    triElements = np.zeros((12*hexElements.shape[0], 3), dtype=int)
    for i in range(hexElements.shape[0]):
        nodes = hexElements[i, :]

        tri1 = np.array([nodes[0], nodes[4], nodes[6]])
        tri2 = np.array([nodes[0], nodes[2], nodes[6]])
        tri3 = np.array([nodes[6], nodes[3], nodes[7]])
        tri4 = np.array([nodes[6], nodes[2], nodes[3]])

        tri5 = np.array([nodes[0], nodes[2], nodes[3]])
        tri6 = np.array([nodes[0], nodes[1], nodes[3]])
        tri7 = np.array([nodes[5], nodes[0], nodes[1]])
        tri8 = np.array([nodes[5], nodes[0], nodes[4]])

        tri9 = np.array([nodes[5], nodes[7], nodes[6]])
        tri10 = np.array([nodes[6], nodes[5], nodes[4]])
        tri11 = np.array([nodes[3], nodes[1], nodes[5]])
        tri12 = np.array([nodes[3], nodes[5], nodes[7]])

        triElements[12*i:12*i+12, :] = np.vstack((tri1, tri2, tri3, tri4, tri5, tri6, tri7, tri8, tri9, tri10, tri11, tri12))
    triElements = np.unique(np.sort(triElements, 1), axis=1)
    trinum = np.argsort(triElements[:, 0])
    triElements = triElements[trinum, :]
    Faces = np.int32(np.hstack((3 * np.ones((triElements.shape[0], 1)), triElements)))
    return triElements, Faces

def PointCloud2Ply(Vertex, Faces, plyname):
    with open(plyname, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0 \n')
        f.write('element vertex '+ str(Vertex.shape[0]) + '\n')
        f.write('property double x\nproperty double y\nproperty double z\n')
        f.write('element face ' + str(Faces.shape[0]) + '\n')
        f.write('property list uchar int vertex_indices\nend_header\n')
        m, n = Vertex.shape
        for i in range(m):
            for j in range(n):
                if j == n-1:
                    f.write(str(Vertex[i, j]) + '\n')
                else:
                    f.write(str(Vertex[i, j]) + '\t')

        m, n = Faces.shape
        for i in range(m):
            for j in range(n):
                if j == n-1:
                    f.write(str(Faces[i, j]) + '\n')
                else:
                    f.write(str(Faces[i, j]) + '\t')

def PointCloud2bou(Boudnum, bouname):
    Boudnum = Boudnum + 1
    with open(bouname, 'w') as f:
        for i in range(Boudnum.shape[0]):
            if ((i + 1) % 20):
                f.write(str(Boudnum[i]) + ' ')
            else:
                f.write(str(Boudnum[i]) + '\n')

def altergravity(file, old_str, gravity):
    """
    modify gravity in .json
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                words = line.split(',')
                words[-2] = ' ' + str(gravity) + ']'
                line = ','.join(words)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

