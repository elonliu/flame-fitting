
import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir


def save_model_joints_info(model, outdir):
    # Save joints info
    j_num = len(model.J)
    with open(outdir + '/vector_joints.txt', 'w') as fout:
        fout.write(str(j_num) + '\n')
        for i in range(j_num):
            for p in range(0, len(model.J[i])):
                string = str(model.J[i][p])
                string = string.replace('[', '')
                string = string.replace(']', '') + ' '
                fout.write(string)
            fout.write('\n')

    #print model.weights
    v_num = len(model.weights)
    with open(outdir + '/vector_weights.txt', 'w') as fout:
        fout.write(str(v_num) + '\n')
        for i in range(v_num):
            fout.write(str(len(model.weights[i])) + '\n')
            for p in range(0, len(model.weights[i])):
                string = str(model.weights[i][p])
                string = string.replace('[', '')
                string = string.replace(']', '') + '\n'
                fout.write(string)
            fout.write('\n')

def save_model_pose_info(model, outname):
    with open(outname, 'w') as f:
        f.write('1 0 0\n')
        f.write('0 1 0\n')
        f.write('0 0 1\n')
        f.write('{} {} {}\n'.format(model.trans.r[0], model.trans.r[1], model.trans.r[2]))
        f.write('{}\n'.format(len(model.pose.r)))
        np.savetxt(f, model.pose.r, fmt='%.8f')

def save_model_exp_info(model, outname):
    with open(outname, 'w') as f:
        f.write('100\n')
        np.savetxt(f, model.betas.r[300:], fmt='%.8f')

def save_model_pose_bs(model, outname):
    bs_ori = model.posedirs.r
    n_dim = bs_ori.shape[2]
    bs = bs_ori.reshape(-1, n_dim)
    n_v = bs.shape[0]
    with open(outname, 'wt') as f:
        f.write('{}\n'.format(n_dim))
        for i in range(n_dim):
            f.write('{}\n'.format(n_v))
            np.savetxt(f, bs[:,i], fmt='%.8f')

def generate_pose_sequence(output_dir):
    # Load FLAME model (here we load the female model)
    # Make sure path is correct
    model_path = './models/generic_model.pkl'
    model = load_model( model_path )           # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path

    safe_mkdir( output_dir )
    safe_mkdir( output_dir + '/gtcoeff' )
    safe_mkdir( output_dir + '/gtcoeff/pose_coeff' )
    safe_mkdir( output_dir + '/gtcoeff/exp_coeff' )

    # Assign random pose and shape parameters
    model.pose[:]  = np.random.randn( model.pose.size ) * 0.0
    model.betas[:] = np.random.randn( model.betas.size ) * 0.0
    
    save_model_joints_info(model, output_dir)
    save_model_pose_bs(model, output_dir+"/pose_bs.txt")

    model.betas[0:300] = np.random.randn( 300 ) * 0.5
    save_model_joints_info(model, output_dir+"/gtcoeff")
    
    # model.trans[:] = np.random.randn( model.trans.size ) * 0.01   # you may also manipulate the translation of mesh
    
    # Save zero pose
    outmesh_path = join( output_dir, '0000.obj' )
    write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
    np.savetxt(output_dir + '/gtcoeff/beta.txt', model.betas.r, fmt='%.8f')
    save_model_pose_info(model, output_dir+"/gtcoeff/pose_coeff/0000.txt")

    # Write to an .obj file
    for idx in range(1, 10):
        model.pose[0:3] = np.random.randn(3) * 0.01
        model.pose[3:6] = np.random.randn(3) * 0.2
        model.pose[6] = abs(np.random.randn(1)) * 0.03
        model.pose[7:9] = np.random.randn(2) * 0.01
        model.trans[:] = np.random.randn( model.trans.size ) * 0.01
        outmesh_path = join( output_dir, '{:04d}.obj'.format(idx) )
        write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
        save_model_pose_info(model, output_dir+"/gtcoeff/pose_coeff/{:04d}.txt".format(idx))
        # Print message
        print 'output mesh saved to: ', outmesh_path 

def generate_exp_sequence(output_dir):
    # Load FLAME model (here we load the female model)
    # Make sure path is correct
    model_path = './models/generic_model.pkl'
    model = load_model( model_path )           # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path

    safe_mkdir( output_dir )
    safe_mkdir( output_dir + '/gtcoeff' )
    safe_mkdir( output_dir + '/gtcoeff/pose_coeff' )
    safe_mkdir( output_dir + '/gtcoeff/exp_coeff' )

    # Assign random pose and shape parameters
    model.pose[:]  = np.random.randn( model.pose.size ) * 0.0
    model.betas[:] = np.random.randn( model.betas.size ) * 0.0
    
    save_model_joints_info(model, output_dir)
    save_model_pose_bs(model, output_dir+"/pose_bs.txt")

    model.betas[0:300] = np.random.randn( 300 ) * 0.5
    save_model_joints_info(model, output_dir+"/gtcoeff")
    
    # model.trans[:] = np.random.randn( model.trans.size ) * 0.01   # you may also manipulate the translation of mesh
    
    # Save zero pose
    outmesh_path = join( output_dir, '0000.obj' )
    write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
    np.savetxt(output_dir + '/gtcoeff/beta.txt', model.betas.r, fmt='%.8f')
    save_model_pose_info(model, output_dir+"/gtcoeff/pose_coeff/0000.txt")
    save_model_exp_info(model, output_dir + '/gtcoeff/exp_coeff/0000.txt')

    # Write to an .obj file
    for idx in range(1, 10):
        model.pose[0:5] = np.random.randn(5) * 0.01
        model.pose[6] = abs(np.random.randn(1)) * 0.3
        model.pose[7:9] = np.random.randn(2) * 0.01
        model.trans[:] = np.random.randn( model.trans.size ) * 0.01
        model.betas[300:] = np.random.randn(100) * 1
        outmesh_path = join( output_dir, '{:04d}.obj'.format(idx) )
        write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
        save_model_pose_info(model, output_dir+"/gtcoeff/pose_coeff/{:04d}.txt".format(idx))
        save_model_exp_info(model, output_dir+"/gtcoeff/exp_coeff/{:04d}.txt".format(idx))

        # Print message
        print 'output mesh saved to: ', outmesh_path 
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    #generate_pose_sequence(R'D:\WorkingItems\neckbuild\dataset\flame_synth\pose_data')
    generate_exp_sequence(R'D:\WorkingItems\neckbuild\dataset\flame_synth\exp_data')