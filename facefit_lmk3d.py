'''
demo: fit FLAME face model to 3D landmarks
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

import numpy as np
import chumpy as ch
from os.path import join

from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir

# -----------------------------------------------------------------------------

def fit_lmk3d( lmk_3d,                      # input landmark 3d
               model,                       # model
               lmk_face_idx, lmk_b_coords,  # landmark embedding
               weights,                     # weights for the objectives
               shape_num=300, expr_num=100, opt_options=None ,
               fix_exp = False):
    """ function: fit FLAME model to 3d landmarks

    input: 
        lmk_3d: input landmark 3d, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        weights: weights for each objective
        shape_num, expr_num: numbers of shape and expression compoenents used
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    # variables
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx )
    if (fix_exp):
        used_idx = shape_idx

    model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero
    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    free_variables = [ model.trans, model.pose, model.betas[used_idx] ] 
    
    # weights
    print "fit_lmk3d(): use the following weights:"
    for kk in weights.keys():
        print "fit_lmk3d(): weights['%s'] = %f" % ( kk, weights[kk] ) 

    # objectives
    # lmk
    lmk_err = landmark_error_3d( mesh_verts=model, 
                                 mesh_faces=model.f, 
                                 lmk_3d=lmk_3d, 
                                 lmk_face_idx=lmk_face_idx, 
                                 lmk_b_coords=lmk_b_coords, 
                                 weight=weights['lmk'] )
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation
    objectives = {}
    objectives.update( { 'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err } ) 

    # options
    if opt_options is None:
        print "fit_lmk3d(): no 'opt_options' provided, use default settings."
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
        opt_options['maxiter'] = 100
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass
        
    # optimize
    # step 1: rigid alignment
    from time import time
    timer_start = time()
    print "\nstep 1: start rigid fitting..."    
    ch.minimize( fun      = lmk_err,
                 x0       = [ model.trans, model.pose[0:3] ],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print "step 1: fitting done, in %f sec\n" % ( timer_end - timer_start )

    # step 2: non-rigid alignment
    timer_start = time()
    print "step 2: start non-rigid fitting..."    
    ch.minimize( fun      = objectives,
                 x0       = free_variables,
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print "step 2: fitting done, in %f sec\n" % ( timer_end - timer_start )

    # return results
    parms = { 'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r }
    return model.r, model.f, parms

# -----------------------------------------------------------------------------

def run_fitting_demo():

    # input landmarks
    lmk_path = './data/landmark_3d.pkl'
    lmk_3d = load_binary_pickle( lmk_path )
    print "loaded 3d landmark from:", lmk_path

    # model
    model_path = './models/male_model.pkl' # change to 'female_model.pkl' or 'generic_model.pkl', if needed
    model = load_model( model_path )       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path

    # landmark embedding
    lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding( lmk_emb_path )
    print "loaded lmk embedding"

    # output
    output_dir = './output'
    safe_mkdir( output_dir )

    # weights
    weights = {}
    weights['lmk']   = 1.0
    weights['shape'] = 0.001
    weights['expr']  = 0.001
    weights['pose']  = 0.1
    
    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 100
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d( lmk_3d=lmk_3d,                                         # input landmark 3d
                                       model=model,                                           # model
                                       lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                       weights=weights,                                       # weights for the objectives
                                       shape_num=300, expr_num=100, opt_options=opt_options ) # options

    # write result
    output_path = join( output_dir, 'fit_lmk3d_result.obj' )
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )

# -----------------------------------------------------------------------------

def run_fitting(lmk_fid_file, lmk_bary_file, lmk_3d_file, output_dir, fix_exp):
    # input landmarks#
    lmk_3d = []
    fin = open(lmk_3d_file, 'r')
    data = fin.read()
    rows = data.split('\n')
    for row in rows:
        split_row = row.split(' ')
        if len(split_row) != 3:
            continue
        for i in range(3):
            split_row[i] = float(split_row[i])
        lmk_3d.append(split_row[0:3])
    fin.close()
    #print lmk_3d
    lmk_3d = np.array(lmk_3d)

    # model
    model_path = './models/male_model.pkl' # change to 'female_model.pkl' or 'generic_model.pkl', if needed
    model = load_model( model_path )       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path
    #print model.J
    #print model.weights


    # landmark embedding
    lmk_face_idx = []
    fin = open(lmk_fid_file, 'r')
    data = fin.read()
    rows = data.split('\n')
    for row in rows:
        if row == '':
            continue
        lmk_face_idx.append(int(row))
    fin.close()
    lmk_face_idx = np.array(lmk_face_idx)

    lmk_b_coords = []
    fin = open(lmk_bary_file, 'r')
    data = fin.read()
    rows = data.split('\n')
    for row in rows:
        split_row = row.split(' ')
        if len(split_row) != 3:
            continue
        for i in range(3):
            split_row[i] = float(split_row[i])
        lmk_b_coords.append(split_row[0:3])
    fin.close()
    lmk_b_coords = np.array(lmk_b_coords)

    # output
    #output_dir = './output'
    safe_mkdir( output_dir )

    # weights
    weights = {}
    weights['lmk']   = 1.0
    weights['shape'] = 0.001
    weights['expr']  = 0.001
    weights['pose']  = 0.1
    
    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 100
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d( lmk_3d=lmk_3d, # input landmark 3d
                                       model=model, # model
                                       lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords, # landmark embedding
                                       weights=weights, # weights for the objectives
                                       shape_num=300, expr_num=100, opt_options=opt_options, fix_exp=fix_exp ) # options

    # write result
    output_path = join( output_dir, 'fit_lmk3d_result.obj' )
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )

    print model.J
    j_num = len(model.J)
    fout = open(output_dir + '/vector_joints.txt', 'w')
    fout.write(str(j_num) + '\n')
    for i in range(j_num):
        for p in range(0, len(model.J[i])):
            string = str(model.J[i][p])
            string = string.replace('[', '')
            string = string.replace(']', '') + ' '
            fout.write(string)
        fout.write('\n')
    fout.close()

    #print model.weights
    v_num = len(model.weights)
    fout = open(output_dir + '/vector_weights.txt', 'w')
    fout.write(str(v_num) + '\n')
    for i in range(v_num):
       fout.write(str(len(model.weights[i])) + '\n')
       for p in range(0, len(model.weights[i])):
           string = str(model.weights[i][p])
           string = string.replace('[', '')
           string = string.replace(']', '') + '\n'
           fout.write(string)
       fout.write('\n')
    fout.close()

    #save model params
    np.savetxt(output_dir + '/beta.txt', model.betas.r, fmt='%.6f')
    np.savetxt(output_dir + '/pose.txt', model.pose.r, fmt='%.6f')

if __name__ == '__main__':
    workdir = R'D:\WorkingItems\neckcap\1208\flame\result\python_flame_exp0'
    run_fitting(workdir + '\\lmk_face_idx.txt', workdir + '\\lmk_b_coords.txt', workdir + '\\landmark_3d.txt', workdir + '\\fitted', fix_exp = True)

