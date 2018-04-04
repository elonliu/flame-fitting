import cPickle as pickle
import numpy as np
import struct
import itertools
import scipy.sparse as sp

def saveMatrix(mat_data_array, num_rows, num_cols, fid):
    assert(num_rows * num_cols == mat_data_array.size)
    mat_data_array = mat_data_array.astype('float32')
    nums = np.array([num_rows, num_cols, mat_data_array.dtype.itemsize], 'int32')
    nums.tofile(fid)
    mat_data_array.tofile(fid)

def saveSparseMatrix(ori_mat, num_rows, num_cols, fid):
    assert(num_rows == ori_mat.shape[0] and num_cols == ori_mat.shape[1])
    cx = ori_mat.tocoo()
    triplets = []
    for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
        new_item = struct.pack('iif', i, j, v)
        triplets.append(new_item)

    assert(len(triplets) == ori_mat.nnz)

    nums = np.array([num_rows, num_cols, ori_mat.nnz, 4], 'int32')
    nums.tofile(fid)
    for t in triplets:
        fid.write(t)


def convertSMPLModel(ori_name, new_name):
    print 'Coverting {} ...'.format(ori_name)
    dd = pickle.load(open(ori_name, 'rb'))

    with open(new_name, 'wb') as f:
        num_verts = dd['v_template'].shape[0] # num_verts
        num_nodes = dd['J'].shape[0] - 1      # num_nodes
        num_beta = dd['shapedirs'].shape[2]  # num_beta
        nums = np.array([num_verts, num_nodes, num_beta], 'int32')
        nums.tofile(f)

        saveMatrix(dd['v_template'], 3*num_verts, 1, f)
        saveMatrix(np.transpose(dd['shapedirs'].r, (2,0,1)), 3*num_verts, num_beta, f)
        saveMatrix(np.transpose(dd['posedirs'], (2,0,1)), 3*num_verts, 9 * num_nodes, f)
        saveSparseMatrix(dd['J_regressor'], num_nodes+1, num_verts, f)
        saveSparseMatrix(sp.coo_matrix(dd['weights'].transpose()), num_nodes+1, num_verts, f)

        num_face = dd['f'].shape[0]
        np.array([num_face], 'int32').tofile(f)
        dd['f'].tofile(f)

        parent = dd['kintree_table'][0,:].astype('int32')
        parent[0] = -1
        for i in range(num_nodes+1):
            saveMatrix(dd['J'][i,:], 3, 1, f)
            saveMatrix(np.eye(4,4), 4, 4, f)
            parent[i].tofile(f)
    print 'Done'

if __name__ == "__main__":
    convertSMPLModel('./models/female_model.pkl', './models/female_model.bin')
    convertSMPLModel('./models/generic_model.pkl', './models/generic_model.bin')
    convertSMPLModel('./models/male_model.pkl', './models/male_model.bin')

